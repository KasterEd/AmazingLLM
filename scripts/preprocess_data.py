# scripts/preprocess_data.py

import argparse
import os
import sys

from datasets import DatasetDict

# Ensure we can import from src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.gpt_response.mcq_dataset import MCQConfig, build_hf_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MCQ CSV data into a HuggingFace DatasetDict."
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to train.csv (with answers).",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test.csv (without answers). Optional.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Fraction of training data used as validation split (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split (default: 42).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = MCQConfig(
        train_path=args.train_csv,
        test_path=args.test_csv,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    print("Building HuggingFace DatasetDict from CSV files...")
    ds: DatasetDict = build_hf_dataset(cfg)

    print("Dataset splits:", list(ds.keys()))
    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} examples")

    print(f"Saving dataset to: {args.output_dir}")
    ds.save_to_disk(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
