#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

NA_TOKEN = "<NA>"
IDK_TOKEN = "<IDK>"


def canonicalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/saq_llama_v1")
    ap.add_argument("--model_dir", default="experiments/exp_2026_01_03_baseline/model")
    ap.add_argument("--out_tsv", default="experiments/exp_2026_01_03_baseline/saq_prediction.tsv")
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load test dataset (prompt-only)
    ds = load_dataset("json", data_files={"test": os.path.join(args.data_dir, "test.jsonl")})["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect if this directory is a PEFT adapter (LoRA)
    adapter_config_path = os.path.join(args.model_dir, "adapter_config.json")
    is_peft = os.path.exists(adapter_config_path)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    if is_peft:
        # Load base model name from adapter config
        peft_cfg = PeftConfig.from_pretrained(args.model_dir)
        base_name = peft_cfg.base_model_name_or_path

        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=dtype,
            device_map=device_map,
        )

        # CRITICAL: resize before loading adapter weights
        base_model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(base_model, args.model_dir)

        # Optional: merge adapter into base for faster inference
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            device_map=device_map,
        )
        # Keep model/tokenizer consistent even for non-PEFT checkpoints
        model.resize_token_embeddings(len(tokenizer))

    model.eval()

    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)

    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ID", "answer"])

        for ex in ds:
            prompt = ex["prompt"]
            ex_id = ex["ID"]

            inputs = tokenizer(prompt, return_tensors="pt", padding=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen = tokenizer.decode(out[0], skip_special_tokens=False)
            # Extract text after the last "Answer:"
            if "Answer:" in gen:
                ans = gen.split("Answer:")[-1]
            else:
                ans = gen[len(prompt):]

            # Stop at newline if present
            ans = ans.split("\n")[0]
            ans = canonicalize(ans)

            # Normalize special tokens and strip any leftover token artifacts
            if NA_TOKEN in ans:
                ans = NA_TOKEN
            elif IDK_TOKEN in ans:
                ans = IDK_TOKEN
            else:
                # remove leading/trailing punctuation that commonly appears in generation
                ans = ans.strip(" \t\r\n\"'`.,;:!?)(")

            w.writerow([ex_id, ans])

    print(f"Wrote predictions to: {args.out_tsv}")


if __name__ == "__main__":
    main()
