#!/bin/bash
# Example: Validate and clean translated dataset

LANG="it"

# Validate translation quality
echo "Validating translation quality..."
python scripts/validate_translation_quality.py \
    --original data/TISER_train.json \
    --translated data/translated/TISER_train_${LANG}.json \
    --language $LANG \
    --sample_size 200 \
    --output data/translated/validation_report_${LANG}.json \
    --export_samples data/translated/problematic_samples_${LANG}.json

# Clean and normalize dataset
echo "Cleaning and normalizing dataset..."
python multilingual_tiser/preprocess/validate_tiser_dataset.py \
    --input data/translated/TISER_train_${LANG}.json \
    --output data/translated/TISER_train_${LANG}_clean.json \
    --split train \
    --prompt_file data/prompts/prompt_${LANG}.txt \
    --save_invalid data/translated/invalid_samples_${LANG}.json

echo "✓ Validation and cleaning complete!"
echo "Clean dataset: data/translated/TISER_train_${LANG}_clean.json"
