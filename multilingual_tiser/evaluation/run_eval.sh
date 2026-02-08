#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

MODEL=$1
LANG=$2
ADAPTER_DIR=$3
MAX_EVAL_SAMPLES=${4:-""}

if [[ -z "$MODEL" || -z "$LANG" || -z "$ADAPTER_DIR" ]]; then
  echo "Usage: bash multilingual/evaluation/run_eval.sh [qwen|mistral] [it|fa|de|en] [path/to/adapter] [max_samples(optional)]"
  exit 1
fi

if [[ "$MODEL" == "qwen" ]]; then
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
elif [[ "$MODEL" == "mistral" ]]; then
  BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
else
  echo "Invalid model type. Use qwen or mistral."
  exit 1
fi

TEST_FILE="data/splits/test/TISER_test_${LANG}.json"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$ADAPTER_DIR/results"

OUT_FILE="${ADAPTER_DIR}/results/eval_${LANG}_${TIMESTAMP}_${MAX_EVAL_SAMPLES:-all}.json"

# Construct command array
CMD=(
  python multilingual_tiser/evaluation/evaluate.py
  --base_model "$BASE_MODEL"
  --adapter_dir "$ADAPTER_DIR"
  --test_file "$TEST_FILE"
  --output_file "$OUT_FILE"
  --max_new_tokens 512
  --use_chat_template
  --only_passed
)

if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
  CMD+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
fi

echo "=========================================="
echo "Running Evaluation"
echo "Base Model: $BASE_MODEL"
echo "Adapter   : $ADAPTER_DIR"
echo "Test File : $TEST_FILE"
echo "=========================================="

"${CMD[@]}"