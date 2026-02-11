#!/bin/bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

MODEL_TYPE=$1       # qwen or mistral
LANG=$2             # it, fa, de, en
ADAPTER_DIR=$3      # Path to the adapter folder
STRATEGY=${4:-base} # base or iterative (defaults to base)
MAX_SAMPLES=${5:-}  

if [[ -z "$MODEL_TYPE" || -z "$LANG" || -z "$ADAPTER_DIR" ]]; then
  echo "Usage: bash run_eval_pipeline.sh [qwen|mistral] [lang] [adapter_path] [base|iterative] [max_samples]"
  exit 1
fi

# Select Base Model
if [[ "$MODEL_TYPE" == "qwen" ]]; then
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
else
  BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
fi

TEST_FILE="data/splits/test/TISER_test_${LANG}.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${ADAPTER_DIR}/results"
mkdir -p "$RESULTS_DIR"

GEN_OUTPUT="${RESULTS_DIR}/gen_${LANG}_${STRATEGY}_${TIMESTAMP}.json"

MAX_SAMPLES_ARG=""
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES_ARG="--max_eval_samples $MAX_SAMPLES"
fi

echo "Strategy: $STRATEGY"

python multilingual_tiser/evaluation/inference.py \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ADAPTER_DIR" \
  --test_file "$TEST_FILE" \
  --output_file "$GEN_OUTPUT" \
  --max_new_tokens 1024 \
  --batch_size 4 \
  --strategy "$STRATEGY" \
  --only_passed \
  --max_extensions 2 \
  $MAX_SAMPLES_ARG

echo "Generation complete: $GEN_OUTPUT"
