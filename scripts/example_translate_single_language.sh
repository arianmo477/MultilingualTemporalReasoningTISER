#!/bin/bash
# Example: Translate TISER dataset to Italian

# Fix for macOS OpenMP library conflict
export KMP_DUPLICATE_LIB_OK=TRUE

# Translate training data (first 1000 samples for testing)
python multilingual_tiser/translation/translate_dataset.py \
    --input data/TISER_train.json \
    --output_dir data/translated \
    --category train \
    --language it \
    --batch_size 8 \
    --max_samples 1000

# Translate test data (first 500 samples for testing)
python multilingual_tiser/translation/translate_dataset.py \
    --input data/TISER_test.json \
    --output_dir data/translated \
    --category test \
    --language it \
    --batch_size 8 \
    --max_samples 500

echo "✓ Italian translation complete!"
echo "Output: data/translated/TISER_train_it.json"
echo "Output: data/translated/TISER_test_it.json"
