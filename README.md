# Secrets Behind the LLMs — Culturally Grounded QA with Parameter-Efficient Fine-Tuning

> A two-pipeline LLM system for culturally grounded question answering, built for strict evaluation formats, reproducible training, and submission-ready inference.

[![Task](https://img.shields.io/badge/Task-MCQ%20%2B%20SAQ-blue)](#project-overview)
[![Fine-tuning](https://img.shields.io/badge/Fine--tuning-LoRA%20%2B%204--bit%20NF4-green)](#technical-highlights)
[![Framework](https://img.shields.io/badge/Framework-HuggingFace-orange)](#tech-stack)
[![Evaluation](https://img.shields.io/badge/Codabench-0.68%20overall-success)](#results)

---

## Project Overview

This project adapts instruction-tuned LLaMA-family models to answer **culturally grounded questions** in two formats:

1. **MCQ — Multiple-Choice Question Answering**  
   Predicts one valid answer among **A, B, C, D** and exports the result as a one-hot table.

2. **SAQ — Short-Answer Question Answering**  
   Generates a concise English answer, or one of two explicit control tokens:
   - `<NA>` — not applicable in the given country
   - `<IDK>` — unknown / insufficient information

The main engineering goal was not only to fine-tune an LLM, but to build a clean, reproducible pipeline that turns messy multilingual/cultural QA data into evaluator-compatible predictions.

---

## Why This Project Stands Out

- Built **two task-specific LLM pipelines** instead of forcing one generic solution.
- Used **LoRA-based parameter-efficient fine-tuning** to adapt LLaMA-family models under limited compute.
- Applied **4-bit NF4 quantization** for the larger MCQ model to reduce VRAM requirements.
- Designed strict output contracts for automated evaluation:
  - JSON-only MCQ answers
  - normalized short-answer strings for SAQ
- Implemented deterministic inference with greedy decoding for reproducible submissions.
- Added robust post-processing to recover valid outputs even when raw generation is imperfect.
- Submitted final predictions to Codabench and analyzed task-level performance differences.

---

## System Architecture

```text
Raw CSV Data
│
├── MCQ Pipeline
│   ├── Normalize A/B/C/D choices
│   ├── Build JSON-only supervision targets
│   ├── Fine-tune LLaMA-3 8B Instruct with LoRA + 4-bit NF4
│   ├── Deterministic JSON generation
│   └── Export one-hot TSV: MCQID, A, B, C, D
│
└── SAQ Pipeline
    ├── Aggregate crowd annotations
    ├── Derive <NA>, <IDK>, or short-answer targets
    ├── Fine-tune LLaMA-3.2 1B Instruct
    ├── Deterministic short-answer generation
    └── Export TSV: ID, answer
```

---

## Results

Final Codabench evaluation results:

| Metric / Split | MCQ | SAQ |
|---|---:|---:|
| Overall Accuracy | — | **0.68** |
| Accuracy | **0.77** | **0.59** |
| China | 0.72 | 0.55 |
| Iran | 0.64 | 0.51 |
| UK | **0.90** | 0.60 |
| US | 0.83 | **0.70** |

**Key observation:** MCQ performed better because it is a constrained classification-style task with only four possible choices. SAQ was harder because it requires exact short-answer generation, where synonyms, formatting, and ambiguity affect evaluation.

---

## Technical Highlights

### MCQ Pipeline

The MCQ task was framed as causal language modeling with a strict JSON output:

```json
{"answer_choice": "A"}
```

Important design choices:

- Normalized all choices into a fixed four-line format.
- Added optional country and language conditioning tags.
- Masked prompt tokens with `-100` so the loss focused only on answer generation.
- Used LoRA adapters on attention and feed-forward projection layers.
- Loaded the base model with 4-bit NF4 quantization for memory-efficient training.
- Used deterministic greedy generation with a small token budget.
- Parsed output through a fallback chain:
  1. JSON parsing
  2. quoted-letter regex
  3. bare-letter regex
  4. conservative fallback

### SAQ Pipeline

The SAQ task required converting noisy crowd annotations into stable supervised targets.

Important design choices:

- Introduced `<NA>` and `<IDK>` as special tokens.
- Used majority voting over annotation categories.
- Applied conservative tie-breaking: `<NA>` > `<IDK>` > answer.
- Canonicalized answer strings through lowercasing, whitespace normalization, and punctuation cleanup.
- Split train/validation by unique question ID to reduce leakage.
- Used short deterministic decoding to avoid verbose or hallucinated answers.

---

## Tech Stack

| Area | Tools / Methods |
|---|---|
| Language | Python |
| LLM Framework | HuggingFace Transformers, Datasets |
| Fine-tuning | LoRA, PEFT-style adapter training |
| Optimization | 4-bit NF4 quantization, masked causal LM loss |
| Models | LLaMA-3 8B Instruct, LLaMA-3.2 1B Instruct |
| Data Formats | CSV, JSONL, TSV |
| Evaluation | Codabench |
| Inference | deterministic greedy decoding, post-processing validators |

---

## Reproducible Workflow

```bash
# 1. Prepare MCQ data
python scripts/preprocess_mcq.py \
  --input data/mcq_train.csv \
  --output artifacts/mcq_dataset

# 2. Train MCQ adapter
python scripts/train_mcq.py \
  --dataset artifacts/mcq_dataset \
  --output models/mcq_lora_adapter

# 3. Generate MCQ submission
python scripts/predict_mcq.py \
  --test data/mcq_test.csv \
  --adapter models/mcq_lora_adapter \
  --output submissions/mcq_predictions.tsv

# 4. Prepare SAQ data
python scripts/preprocess_saq.py \
  --input data/saq_train.csv \
  --output artifacts/saq_dataset

# 5. Train SAQ model / adapter
python scripts/train_saq.py \
  --dataset artifacts/saq_dataset \
  --output models/saq_model

# 6. Generate SAQ submission
python scripts/predict_saq.py \
  --test data/saq_test.csv \
  --model models/saq_model \
  --output submissions/saq_predictions.tsv
```

> Note: adjust script names and paths to match the final repository layout.

---

## Example Output Formats

### MCQ submission

| MCQID | A | B | C | D |
|---|---:|---:|---:|---:|
| example_001 | 0 | 1 | 0 | 0 |
| example_002 | 1 | 0 | 0 | 0 |

### SAQ submission

| ID | answer |
|---|---|
| example_001 | public holiday |
| example_002 | `<IDK>` |
| example_003 | `<NA>` |

---

## What I Learned

This project strengthened practical skills in:

- turning ambiguous task definitions into strict model-output contracts;
- building robust preprocessing for multilingual and culturally grounded QA data;
- applying parameter-efficient LLM fine-tuning under compute constraints;
- reducing hallucination risk through control tokens and conservative target construction;
- designing deterministic inference pipelines for automated leaderboard evaluation;
- analyzing why classification-style LLM tasks and generative exact-match tasks behave differently.

---

## Limitations and Future Work

Current limitations:

- SAQ exact-match evaluation is sensitive to synonyms and formatting differences.
- Conservative `<NA>` / `<IDK>` tie-breaking can reduce recall for borderline answerable cases.
- Prompt-template mismatch between training and inference may affect calibration.

Planned improvements:

- unify prompt templates across training and inference;
- add constrained decoding or logits masking for MCQ;
- make SAQ target selection deterministic using highest-vote canonical answers;
- improve normalization for numbers, dates, named entities, and culturally specific expressions;
- add error analysis dashboards by country and question type.

---

## Repository Focus

This repository demonstrates an end-to-end LLM engineering workflow:

```text
Data preprocessing → supervised target construction → efficient fine-tuning → deterministic inference → evaluator-ready submissions → result analysis
```

It is designed to be understandable for researchers, engineers, and recruiters reviewing practical LLM fine-tuning experience.

---

## Authors

- **Štepánka Lanková** — Computational Modeling and Simulation, TU Dresden
- **Kaster Kumarbek** — Computational Modeling and Simulation, TU Dresden

---

## Detailed Report

A full technical report is included separately and explains the dataset design, preprocessing logic, training configuration, inference strategy, evaluation results, and future directions in more detail.
