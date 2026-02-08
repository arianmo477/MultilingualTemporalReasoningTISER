#!/bin/bash
# Example: Analyze dataset statistics

# Create output directory if it doesn't exist
mkdir -p data/translated

# Check if translated files exist
if [ -f "data/translated/TISER_train_it_clean.json" ] && [ -f "data/translated/TISER_train_fa_clean.json" ]; then
    echo "Analyzing all languages (EN, IT, FA)..."
    python scripts/analyze_dataset_statistics.py \
        --datasets \
            data/TISER_train.json \
            data/translated/TISER_train_it_clean.json \
            data/translated/TISER_train_fa_clean.json \
        --languages en it fa \
        --output data/translated/statistics_comparison.json
else
    echo "Translated files not found. Analyzing English only..."
    echo "Run ./scripts/example_translate_single_language.sh first to generate translations."
    python scripts/analyze_dataset_statistics.py \
        --datasets data/TISER_train.json \
        --languages en \
        --output data/translated/statistics_en.json
fi

echo ""
echo "✓ Statistics analysis complete!"
