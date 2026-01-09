#!/usr/bin/env python3
import argparse
import ast
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


NA_TOKEN = "<NA>"
IDK_TOKEN = "<IDK>"


def _detect_sep(path: str) -> str:
    if path.endswith(".tsv"):
        return "\t"
    return ","


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    sep = _detect_sep(path)
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def canonicalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'`.,;:!?)(")
    return s


def safe_literal_eval(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def build_prompt(row: Dict[str, str]) -> str:
    # Keep both original question (possibly non-English) and en_question.
    # Country is explicit and early (to force locale conditioning).
    return (
        "You answer short-answer questions.\n"
        "Return a short answer in English.\n"
        f"If the question is not applicable in the given country, output {NA_TOKEN}.\n"
        f"If you do not know, output {IDK_TOKEN}.\n\n"
        f"ID: {row.get('ID','')}\n"
        f"country: {row.get('country','')}\n"
        f"question: {row.get('question','')}\n"
        f"en_question: {row.get('en_question','')}\n"
        "Answer:"
    )


@dataclass
class LabelDecision:
    label: str                       # either NA_TOKEN, IDK_TOKEN, or an English short answer
    label_type: str                  # "NA" | "IDK" | "ANSWER"
    answer_distribution: Dict[str, float]
    votes: Dict[str, int]


def decide_label_from_annotations(
    annotations_raw: str,
    idks_raw: str,
    rng: random.Random,
) -> LabelDecision:
    annotations = safe_literal_eval(annotations_raw)
    idks = safe_literal_eval(idks_raw) or {}

    # Parse idks dict counts
    na_votes = int(idks.get("not-applicable", 0) or 0)
    idk_votes = int(idks.get("idk", 0) or 0)
    no_answer_votes = int(idks.get("no-answer", 0) or 0)

    # Parse answer clusters
    # annotations is expected to be: list[{'en_answers': [...], 'count': k}, ...]
    answer_weights: Dict[str, float] = defaultdict(float)
    answer_votes_total = 0

    if isinstance(annotations, list):
        for cluster in annotations:
            if not isinstance(cluster, dict):
                continue
            cnt = int(cluster.get("count", 0) or 0)
            en_answers = cluster.get("en_answers", [])
            if not en_answers or cnt <= 0:
                continue
            # distribute cluster weight across variants to avoid over-rewarding multi-variant clusters
            per = cnt / max(1, len(en_answers))
            for a in en_answers:
                if not a:
                    continue
                ca = canonicalize_answer(str(a))
                if not ca:
                    continue
                answer_weights[ca] += per
            answer_votes_total += cnt

    # Aggregate "unknown" votes as IDK-family for baseline
    idk_family_votes = idk_votes + no_answer_votes

    votes = {
        "NA": na_votes,
        "IDK": idk_family_votes,
        "ANSWER": answer_votes_total,
    }

    # Majority decision with conservative tie-breaking:
    # NA > IDK > ANSWER in ties (reduces hallucinations)
    best = max(votes.items(), key=lambda kv: (kv[1], {"NA": 2, "IDK": 1, "ANSWER": 0}[kv[0]]))[0]

    if best == "NA" and na_votes > 0:
        return LabelDecision(label=NA_TOKEN, label_type="NA", answer_distribution=dict(answer_weights), votes=votes)

    if best == "IDK" and idk_family_votes > 0:
        return LabelDecision(label=IDK_TOKEN, label_type="IDK", answer_distribution=dict(answer_weights), votes=votes)

    # Otherwise sample an answer from distribution (count-weighted)
    if not answer_weights:
        # Fallback if annotation missing
        return LabelDecision(label=IDK_TOKEN, label_type="IDK", answer_distribution={}, votes=votes)

    items = list(answer_weights.items())
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    chosen = items[-1][0]
    for ans, w in items:
        acc += w
        if acc >= r:
            chosen = ans
            break

    return LabelDecision(label=chosen, label_type="ANSWER", answer_distribution=dict(answer_weights), votes=votes)


def split_by_id(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    ids = sorted(df["ID"].unique().tolist())
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_ratio))) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])
    val_df = df[df["ID"].isin(val_ids)].copy()
    train_df = df[~df["ID"].isin(val_ids)].copy()
    return train_df, val_df


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed/saq_llama_v1")
    ap.add_argument("--train_file", default="train_dataset_saq.csv",
                    help="Filename inside raw_dir (tsv/csv/jsonl).")
    ap.add_argument("--test_file", default="test_dataset_saq.csv",
                    help="Filename inside raw_dir (tsv/csv/jsonl).")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    train_path = os.path.join(args.raw_dir, args.train_file)
    test_path = os.path.join(args.raw_dir, args.test_file)

    train_df = _read_table(train_path)
    test_df = _read_table(test_path)

    # Minimal column checks
    required_train = {"ID", "question", "en_question", "annotations", "idks", "country"}
    required_test = {"ID", "question", "en_question", "country"}
    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)
    if missing_train:
        raise ValueError(f"Missing train columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing test columns: {missing_test}")

    # Split by ID (prevents leakage across locales for the same underlying question ID)
    train_split_df, val_split_df = split_by_id(train_df, args.val_ratio, args.seed)

    stats = {
        "train_rows": int(len(train_df)),
        "train_split_rows": int(len(train_split_df)),
        "val_split_rows": int(len(val_split_df)),
        "test_rows": int(len(test_df)),
        "label_type_counts": {"train": Counter(), "val": Counter()},
        "country_counts": {"train": Counter(), "val": Counter(), "test": Counter()},
    }

    def build_sft_rows(df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
        out = []
        for _, r in df.iterrows():
            row = {k: (r[k] if k in r else "") for k in df.columns}
            prompt = build_prompt(row)

            decision = decide_label_from_annotations(
                annotations_raw=row.get("annotations", ""),
                idks_raw=row.get("idks", ""),
                rng=rng,
            )

            stats["label_type_counts"][split][decision.label_type] += 1
            stats["country_counts"][split][row.get("country", "")] += 1

            out.append({
                "ID": row.get("ID", ""),
                "country": row.get("country", ""),
                "prompt": prompt,
                "target": decision.label,
                "target_type": decision.label_type,
            })
        return out

    train_rows = build_sft_rows(train_split_df, "train")
    val_rows = build_sft_rows(val_split_df, "val")

    # Test rows are prompt-only (no labels)
    test_rows = []
    for _, r in test_df.iterrows():
        row = {k: (r[k] if k in r else "") for k in test_df.columns}
        stats["country_counts"]["test"][row.get("country", "")] += 1
        test_rows.append({
            "ID": row.get("ID", ""),
            "country": row.get("country", ""),
            "prompt": build_prompt(row),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_rows)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test_rows)

    # Metadata
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote:\n- {args.out_dir}/train.jsonl\n- {args.out_dir}/val.jsonl\n- {args.out_dir}/test.jsonl\n- {args.out_dir}/meta.json")


if __name__ == "__main__":
    main()
