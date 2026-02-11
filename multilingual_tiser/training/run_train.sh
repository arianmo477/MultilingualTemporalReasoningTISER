#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

MODEL=$1        # qwen
LANG=$2         # it
MAX_SAMPLES=${3:-}  # 1000

if [[ -z "$MODEL" || -z "$LANG" ]]; then
  echo "Usage: bash multilingual_tiser/training/run_train.sh qwen it 1000"
  exit 1
fi

MAX_SAMPLES_ARG=""
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES_ARG="--max_train_samples $MAX_SAMPLES"
fi

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE="data/splits/train/TISER_train_${LANG}.json"
OUT_DIR="experiments/${MODEL}/${LANG}_${MAX_SAMPLES:-full}_4090_Optimized"
mkdir -p "$OUT_DIR"

# --- RTX 4090 (24GB) SETTINGS ---
EPOCHS=2
LR=2e-4

# 3072 is safe for 24GB VRAM with gradient checkpointing
MAX_LEN=3072

# LoRA Config
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

# Batch Size Strategy for 24GB
# BS=4 per device * 4 accum steps = Effective Batch Size 16
PER_DEVICE_BS=2
GRAD_ACCUM=8

echo "=============================="
echo " STARTING RTX 4090 TRAINING"
echo "Model         : $MODEL_NAME"
echo "Language      : $LANG"
echo "Max Length    : $MAX_LEN"
echo "LoRA Rank     : $LORA_R"
echo "Batch Size    : $PER_DEVICE_BS (Accum: $GRAD_ACCUM)"
echo "=============================="

python multilingual_tiser/training/train_qlora.py \
  --model_name "$MODEL_NAME" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --max_length "$MAX_LEN" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --per_device_batch_size "$PER_DEVICE_BS" \
  --grad_accum "$GRAD_ACCUM" \
  --save_steps 200 \
  --logging_steps 10 \
  --dataloader_num_workers 4 \
  --gradient_checkpointing 1 \
  $MAX_SAMPLES_ARG \
  --only_passed