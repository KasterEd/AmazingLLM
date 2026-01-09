# src/data/mcq_dataset.py

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict


ANSWER_LETTERS = ["A", "B", "C", "D"]


@dataclass
class MCQConfig:
    train_path: str
    test_path: Optional[str] = None
    eval_ratio: float = 0.1
    seed: int = 42


def _parse_choices_json(choices_str: str) -> Dict[str, str]:
    """
    Parse the JSON in the `choices` column.

    Expected format (example):
        {"A": "17", "B": "2", "C": "5", "D": "8"}
    """
    if isinstance(choices_str, dict):
        return choices_str  # already parsed

    try:
        return json.loads(choices_str)
    except Exception as e:
        raise ValueError(f"Failed to parse choices JSON: {choices_str!r}") from e


def _format_choices_block(choices: Dict[str, str]) -> str:
    """
    Format choices dict into a human-readable multi-line block:

        A. 17
        B. 2
        C. 5
        D. 8
    """
    lines = []
    for letter in ANSWER_LETTERS:
        if letter in choices:
            lines.append(f"{letter}. {choices[letter]}")
    return "\n".join(lines)


def _extract_answer_letter(raw_answer) -> Optional[str]:
    """
    Try to extract the answer letter (A/B/C/D) from whatever is in the train CSV.

    Supported formats:
    - "B"
    - " b "
    - '{"answer_choice": "B"}'
    - '{"answer_choice":"b"}'
    """
    if raw_answer is None or (isinstance(raw_answer, float) and np.isnan(raw_answer)):
        return None

    # Already a simple string like "B"
    if isinstance(raw_answer, str):
        stripped = raw_answer.strip().upper()

        # JSON case
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                val = data.get("answer_choice", None)
                if isinstance(val, str):
                    val = val.strip().upper()
                    if val in ANSWER_LETTERS:
                        return val
            except Exception:
                pass

        # Plain letter case
        if stripped in ANSWER_LETTERS:
            return stripped

    # Fallback: nothing found
    return None


def _build_input_and_target(
    df: pd.DataFrame,
    has_answers: bool,
) -> pd.DataFrame:
    """
    Add `input_text`, `target_text`, `answer_letter` (if available) columns.

    Expected columns in df:
    - "prompt"
    - "choices" (JSON string)
    - optionally "answer" OR something equivalent
    - optionally "country", "locale", etc. (we will use "country" if available)
    """
    # Parse choices JSON into Python dict
    df = df.copy()
    df["choices_dict"] = df["choices"].apply(_parse_choices_json)

    # Build the formatted choices string
    df["choices_block"] = df["choices_dict"].apply(_format_choices_block)

    # Extract answer letter if we have answers
    if has_answers:
        answer_col = None
        # Try to guess the answer column name
        for col in df.columns:
            if col.lower() == "answer_idx":
                answer_col = col
                break

        if answer_col is None:
            raise KeyError(
                "No answer column found in train CSV. "
                "Expected a column named 'answer'."
            )

        df["answer_letter"] = df[answer_col].apply(_extract_answer_letter)
        if df["answer_letter"].isnull().any():
            missing = df[df["answer_letter"].isnull()]
            raise ValueError(
                "Some rows in train CSV do not have a valid answer letter "
                "(A/B/C/D) after parsing.\n"
                f"Example bad rows:\n{missing.head()}"
            )
    else:
        df["answer_letter"] = None

    # Country info (if present) – optional
    def get_country(row) -> str:
        if "country" in row and isinstance(row["country"], str) and row["country"].strip():
            return row["country"].strip()
        return ""

    df["country_str"] = df.apply(get_country, axis=1)

    # Build input_text and target_text
    input_texts = []
    target_texts = []

    for _, row in df.iterrows():
        prompt = str(row["prompt"]).strip()
        choices_block = row["choices_block"]
        country = row["country_str"]

        # Optional: prepend country context if non-empty
        if country:
            system_prefix = f"Country context: {country}\n\n"
        else:
            system_prefix = ""

        # Input to the model
        input_text = (
            f"{system_prefix}"
            f"{prompt}\n\n"
            f"Choices:\n{choices_block}\n\n"
            "Respond ONLY with a JSON object of the form:\n"
            '{"answer_choice":"<LETTER>"}\n'
            "where <LETTER> is exactly one of A, B, C, or D."
        )

        input_texts.append(input_text)

        # Target (what we want the model to generate)
        if has_answers:
            letter = row["answer_letter"]
            target_obj = {"answer_choice": letter}
            target_text = json.dumps(target_obj)
        else:
            target_text = ""  # no target in test

        target_texts.append(target_text)

    df["input_text"] = input_texts
    df["target_text"] = target_texts

    return df


def load_mcq_dataframes(cfg: MCQConfig) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load train (+ optional test) CSV into pandas DataFrames and build
    `input_text` and `target_text`.
    """
    train_df = pd.read_csv(cfg.train_path)
    train_df = _build_input_and_target(train_df, has_answers=True)

    test_df: Optional[pd.DataFrame] = None
    if cfg.test_path is not None:
        test_df = pd.read_csv(cfg.test_path)
        test_df = _build_input_and_target(test_df, has_answers=False)

    return train_df, test_df


def build_hf_dataset(cfg: MCQConfig) -> DatasetDict:
    """
    Build a HuggingFace DatasetDict with splits:
    - train
    - validation
    - (optional) test

    Each split has at least:
    - input_text (str)
    - target_text (str)
    - answer_letter (str or None for test)
    - mcqid, id, country (if present in CSV)
    """
    train_df, test_df = load_mcq_dataframes(cfg)

    # Create HF Dataset from train DataFrame
    full_train_ds = Dataset.from_pandas(train_df, preserve_index=False)

    # Split into train/validation
    split = full_train_ds.train_test_split(
        test_size=cfg.eval_ratio, seed=cfg.seed, shuffle=True
    )
    train_ds = split["train"]
    val_ds = split["test"]

    dataset_dict = {"train": train_ds, "validation": val_ds}

    # Test dataset (if provided)
    if test_df is not None:
        test_ds = Dataset.from_pandas(test_df, preserve_index=False)
        dataset_dict["test"] = test_ds

    return DatasetDict(dataset_dict)
