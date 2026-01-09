# src/training/train_mcq.py

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import yaml
from datasets import load_from_disk


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig

# Make src importable when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# -----------------------------
# Data collator for causal LM
# -----------------------------
@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # If there are no labels (e.g. test set during predict), just return inputs
        if "labels" not in features[0]:
            return batch

        # Otherwise, pad labels as before
        labels = [f["labels"] for f in features]
        max_len = batch["input_ids"].shape[1]
        labels_batch = torch.full(
            (len(labels), max_len),
            -100,
            dtype=torch.long,
        )
        for i, l in enumerate(labels):
            l_tensor = torch.tensor(l, dtype=torch.long)
            labels_batch[i, : l_tensor.shape[0]] = l_tensor

        batch["labels"] = labels_batch
        return batch


# -----------------------------
# Config utilities
# -----------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LLaMA-based MCQ assistant with LoRA."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for this experiment.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to preprocessed dataset directory (saved with save_to_disk).",
    )
    return parser.parse_args()


# -----------------------------
# Tokenization / preprocessing
# -----------------------------

def make_tokenize_function(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
):
    """
    Returns a function that:
      - tokenizes `input_text` and `target_text` separately
      - concatenates them
      - masks prompt tokens in labels with -100
    """

    def tokenize_example(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example["input_text"]
        answer = example["target_text"]

        # For test set, target_text may be empty; skip those for training
        if not isinstance(answer, str):
            answer = ""
        if answer.strip() == "":
            # For test data, we do not need labels; just encode the prompt
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            # Truncate from the left if too long
            if len(prompt_ids) + 1 > max_length:
                excess = len(prompt_ids) + 1 - max_length
                prompt_ids = prompt_ids[excess:]

            input_ids = prompt_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # No labels for test set (they will be ignored by Trainer)
            }

        # Training / validation: we have an answer
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
        )["input_ids"]

        # +1 for EOS
        total_len = len(prompt_ids) + len(answer_ids) + 1
        if total_len > max_length:
            # We prefer keeping the answer intact, truncate the prompt from the left
            excess = total_len - max_length
            if excess >= len(prompt_ids):
                # In extreme case, cut prompt completely and keep only answer
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[excess:]

        input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        # labels: mask prompt tokens with -100, train only on answer + EOS
        labels = (
            [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return tokenize_example


# -----------------------------
# Main training routine
# -----------------------------

def main() -> None:
    import transformers
    from transformers import TrainingArguments
    import inspect

    print("TRANSFORMERS VERSION IN SCRIPT:", transformers.__version__)
    print("TrainingArguments module:", TrainingArguments.__module__)
    print("TrainingArguments signature:", inspect.signature(TrainingArguments.__init__))
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # ---- Load dataset ----
    print(f"Loading preprocessed dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)
    print("Splits:", list(dataset.keys()))

    # ---- Load tokenizer ----
    model_name = cfg["model"]["base_model_name"]
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    # Ensure pad token exists (for LLaMA-like models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Tokenize / preprocess dataset ----
    max_length = cfg["data"]["max_length"]

    tokenize_fn = make_tokenize_function(tokenizer, max_length)

    print("Tokenizing training split...")
    tokenized_train = dataset["train"].map(
        tokenize_fn,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train",
    )

    print("Tokenizing validation split...")
    tokenized_val = dataset["validation"].map(
        tokenize_fn,
        remove_columns=dataset["validation"].column_names,
        desc="Tokenizing validation",
    )

    tokenized_test = None
    if "test" in dataset:
        print("Tokenizing test split...")
        tokenized_test = dataset["test"].map(
            tokenize_fn,
            remove_columns=dataset["test"].column_names,
            desc="Tokenizing test",
        )

    # ---- Load base model with 4-bit quantization (optional, but recommended) ----
    # You can disable 4-bit by removing quantization_config and prepare_model_for_kbit_training.
    print(f"Loading base model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    # ---- Apply LoRA ----
    lora_config = LoraConfig(
        r=cfg["model"]["lora_r"],
        lora_alpha=cfg["model"]["lora_alpha"],
        lora_dropout=cfg["model"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training arguments ----
    output_dir = cfg["output"]["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg["training"]["num_epochs"]),
        per_device_train_batch_size=int(cfg["training"]["batch_size"]),
        gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_ratio=float(cfg["training"]["warmup_ratio"]),
        logging_steps=int(cfg["training"]["logging_steps"]),
        eval_strategy="steps",  # see note below
        eval_steps=int(cfg["training"]["eval_steps"]),
        save_steps=int(cfg["training"]["save_steps"]),
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        tf32=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    # ---- Data collator ----
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # ---- Metric function (optional basic loss reporting) ----
    # For now, we just rely on loss; you can add accuracy over answer letters later.
    def compute_metrics(eval_pred):
        # eval_pred is (logits, labels) but for now we just let Trainer log loss
        return {}

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---- Train ----
    print(
        "LR type right before TrainingArguments:",
        type(cfg["training"]["learning_rate"]),
        "value:", cfg["training"]["learning_rate"],
    )
    print(
        "WD type right before TrainingArguments:",
        type(cfg["training"]["weight_decay"]),
        "value:", cfg["training"]["weight_decay"],
    )

    print("Starting training...")
    trainer.train()

    # ---- Save final adapter ----
    print(f"Saving final adapter model to {output_dir}")
    trainer.save_model(output_dir)

    # ---- Optionally, save test predictions (for later analysis / submission) ----
    if tokenized_test is not None:
        print("Running prediction on test split (no labels)...")
        preds = trainer.predict(tokenized_test, metric_key_prefix="test")
        # preds.predictions shape: (num_examples, seq_len, vocab_size)
        # You can add code here to decode predictions if you want.

    print("Training finished.")


if __name__ == "__main__":
    main()
