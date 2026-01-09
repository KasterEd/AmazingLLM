#!/bin/bash
#SBATCH --job-name=train_amazing_LLM_hehe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kaster.kumarbek@mailbox.tu-dresden.de
#SBATCH -o gpt_res_%j.out
#SBATCH -e gpt_res_%j.err
#SBATCH --gres=gpu:1

source .env/bin/activate

#hf auth login --token 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

python experiments/exp_2026_01_03_baseline/predict.py \
  --data_dir data/processed/saq_llama_v1 \
  --model_dir experiments/exp_2026_01_03_baseline/model \
  --out_tsv experiments/exp_2026_01_03_baseline/saq_prediction.tsv