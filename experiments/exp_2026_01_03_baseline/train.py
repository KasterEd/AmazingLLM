#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


NA_TOKEN = "<NA>"
IDK_TOKEN = "<IDK>"


@dataclass
class DataCollatorForSFT:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        # Pad labels to same length
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def build_tokenized_dataset(tokenizer, dataset, max_len: int):
    def tokenize_one(ex):
        prompt = ex["prompt"]
        target = ex["target"]
        # Ensure a single-space separation after "Answer:" then the target.
        full = prompt + " " + target + "\n"

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_enc = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_len)

        input_ids = full_enc["input_ids"]
        attn = full_enc["attention_mask"]

        # Mask prompt tokens; only supervise the answer tokens.
        labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[min(len(prompt_ids), len(input_ids)):]
        labels = labels[:len(input_ids)]

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    return dataset.map(tokenize_one, remove_columns=dataset.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/saq_llama_v1")
    ap.add_argument("--exp_dir", default="experiments/exp_2026_01_03_baseline")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per_device_train_bs", type=int, default=2)
    ap.add_argument("--per_device_eval_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA if peft is installed.")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)

    model_out = os.path.join(args.exp_dir, "model")
    os.makedirs(model_out, exist_ok=True)

    # Load datasets
    data_files = {
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "validation": os.path.join(args.data_dir, "val.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Ensure PAD token exists (common for LLaMA)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Add special control tokens to avoid them being split unpredictably
    special_added = tokenizer.add_special_tokens({"additional_special_tokens": [NA_TOKEN, IDK_TOKEN]})

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if special_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Optional LoRA
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed but --use_lora was set. Install peft or disable --use_lora.")
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules="all-linear",  # broad default; adjust if needed for your model
        )
        model = get_peft_model(model, lora)

    # Tokenize
    train_tok = build_tokenized_dataset(tokenizer, ds["train"], args.max_len)
    val_tok = build_tokenized_dataset(tokenizer, ds["validation"], args.max_len)

    # Training args
    fp16 = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=model_out,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_bs,
        per_device_eval_batch_size=args.per_device_eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        warmup_ratio=0.03,
        weight_decay=0.0,
        fp16=fp16,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DataCollatorForSFT(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save final model + tokenizer
    trainer.save_model(model_out)
    tokenizer.save_pretrained(model_out)
    print(f"Saved model to: {model_out}")


if __name__ == "__main__":
    main()
