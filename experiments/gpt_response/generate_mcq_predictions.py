import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk

# ================== CONFIG =====================
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# LoRA adapter / checkpoint directory (from your training run)
ADAPTER_DIR = Path("experiments/exp_2025-12-09_baseline/checkpoints")

# Path to the *test* dataset folder with arrow/json files
TEST_DATASET_PATH = Path("data/processed/mcq_llama_v1/test")

# Output TSV file
OUT_TSV = ADAPTER_DIR.parent / "mcq_prediction.tsv"
# ==============================================


def load_trained_model():
    """
    Load base LLaMA model + LoRA adapter + tokenizer.
    """
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))

    # Use tokenizer from adapter dir (you likely saved it there)
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.eval()
    return model, tokenizer


def normalize_choices(choices_field):
    """
    The 'choices' field might be:
      - a JSON string: '{"A": "...", "B": "...", ...}'
      - a Python dict already: {"A": "...", "B": "...", ...}
    This function returns a dict.
    """
    if isinstance(choices_field, dict):
        return choices_field
    if isinstance(choices_field, str):
        return json.loads(choices_field)
    # If it's something else (e.g. list of pairs), adapt as needed:
    raise ValueError(f"Unsupported choices format: {type(choices_field)}")


def build_prompt(prompt_text: str, choices_field) -> str:
    """
    Build the same prompt style used for training.
    We assume a dict with keys A,B,C,D.
    """
    choices = normalize_choices(choices_field)

    # Order choices by key (A, B, C, D)
    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

    user_part = (
        f"{prompt_text.strip()}\n\n"
        f"Choices:\n{choices_text}\n\n"
        'Return ONLY JSON with the key "answer_choice", e.g.: {"answer_choice": "B"}.'
    )

    full_text = f"<|user|>\n{user_part}\n<|assistant|>\n"
    return full_text


def extract_choice_from_output(output_text: str) -> str:
    """
    Extract predicted choice letter from model output.
    Expected something like: {"answer_choice": "B"}.
    """
    text = output_text.strip()

    # 1) Try JSON parsing
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            parsed = json.loads(json_str)
            val = parsed.get("answer_choice", "").strip()
            if val in ["A", "B", "C", "D"]:
                return val
    except Exception:
        pass

    # 2) Regex for "A"/"B"/"C"/"D" inside quotes
    m = re.search(r'"([ABCD])"', text)
    if m:
        return m.group(1)

    # 3) Bare letter A/B/C/D
    m = re.search(r'\b([ABCD])\b', text)
    if m:
        return m.group(1)

    # 4) Fallback if nothing is found
    return "A"


def predict_for_example(model, tokenizer, prompt_text: str, choices_field) -> str:
    """
    Run the model for one MCQ and return predicted choice letter.
    """
    text = build_prompt(prompt_text, choices_field)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    # Take only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    choice = extract_choice_from_output(decoded)
    return choice


def main():
    # 1. Load HF test dataset
    print(f"Loading test dataset from: {TEST_DATASET_PATH}")
    test_ds = load_from_disk(str(TEST_DATASET_PATH))

    print("Test dataset columns:", test_ds.column_names)

    # We assume these columns exist; adjust names if necessary.
    # At minimum, we need MCQID, prompt, choices.
    required_cols = {"MCQID", "prompt", "choices"}
    missing = required_cols - set(test_ds.column_names)
    if missing:
        raise ValueError(f"Missing columns in test dataset: {missing}")

    # 2. Load model + tokenizer
    model, tokenizer = load_trained_model()

    predictions = []

    # 3. Iterate through dataset examples
    for idx, example in enumerate(test_ds):
        mcqid = example["MCQID"]
        prompt_text = example["prompt"]
        choices_field = example["choices"]

        choice = predict_for_example(model, tokenizer, prompt_text, choices_field)

        row = {
            "MCQID": mcqid,
            "A": (choice == "A"),
            "B": (choice == "B"),
            "C": (choice == "C"),
            "D": (choice == "D"),
        }
        predictions.append(row)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} examples...")

    # 4. Save to mcq_prediction.tsv
    df = pd.DataFrame(predictions, columns=["MCQID", "A", "B", "C", "D"])
    print(f"Writing predictions to: {OUT_TSV}")
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)

    print("Done. First few lines:")
    print(df.head())


if __name__ == "__main__":
    main()
