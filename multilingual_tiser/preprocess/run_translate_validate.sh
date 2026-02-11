#!/usr/bin/env bash
set -euo pipefail

# Args
lang=${1:-it}      # e.g., it, fa, de
category=${2:-train} # e.g., train, test

# Files
TARGET="data/splits/${category}/TISER_${category}_${lang}.json"
SOURCE="data/splits/${category}/TISER_${category}_en.json"

echo "Running Validation + Scoring + Fixing..."
python3 multilingual_tiser/preprocess/translate_validate.py \
    --input "$TARGET" \
    --en "$SOURCE" \
    --output "$TARGET" \
    --category "$category"

echo "Done."