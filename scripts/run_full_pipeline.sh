#!/bin/bash
# Full pipeline to translate and validate TISER datasets for all languages

set -e  # Exit on error

echo "=============================================="
echo "TISER Multilingual Translation Pipeline"
echo "=============================================="

# Configuration
DATA_DIR="data"
OUTPUT_DIR="data/translated"
SCRIPTS_DIR="scripts"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Languages to process
LANGUAGES=("it" "fa" "de")

echo ""
echo "Step 1: Translating datasets..."
echo "=============================================="

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "Processing language: $lang"
    
    # Translate training data
    echo "  - Translating training data..."
    python multilingual_tiser/translation/translate_dataset.py \
        --input "$DATA_DIR/TISER_train.json" \
        --output_dir "$OUTPUT_DIR" \
        --category train \
        --language "$lang" \
        --batch_size 8 \
        --max_samples 0
    
    # Translate test data
    echo "  - Translating test data..."
    python multilingual_tiser/translation/translate_dataset.py \
        --input "$DATA_DIR/TISER_test.json" \
        --output_dir "$OUTPUT_DIR" \
        --category test \
        --language "$lang" \
        --batch_size 8 \
        --max_samples 0
    
    echo "  ✓ Translation complete for $lang"
done

echo ""
echo "Step 2: Validating translated datasets..."
echo "=============================================="

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "Validating $lang translations..."
    
    # Validate training data
    python "$SCRIPTS_DIR/validate_translation_quality.py" \
        --original "$DATA_DIR/TISER_train.json" \
        --translated "$OUTPUT_DIR/TISER_train_${lang}.json" \
        --language "$lang" \
        --sample_size 200 \
        --output "$OUTPUT_DIR/validation_report_train_${lang}.json" \
        --export_samples "$OUTPUT_DIR/problematic_samples_train_${lang}.json"
    
    echo "  ✓ Training validation complete"
done

echo ""
echo "Step 3: Preprocessing validated datasets..."
echo "=============================================="

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "Preprocessing $lang datasets..."
    
    # Validate and normalize training data
    python multilingual_tiser/preprocess/validate_tiser_dataset.py \
        --input "$OUTPUT_DIR/TISER_train_${lang}.json" \
        --output "$OUTPUT_DIR/TISER_train_${lang}_clean.json" \
        --split train \
        --prompt_file "data/prompts/prompt_${lang}.txt" \
        --save_invalid "$OUTPUT_DIR/invalid_samples_train_${lang}.json"
    
    # Validate and normalize test data
    python multilingual_tiser/preprocess/validate_tiser_dataset.py \
        --input "$OUTPUT_DIR/TISER_test_${lang}.json" \
        --output "$OUTPUT_DIR/TISER_test_${lang}_clean.json" \
        --split test \
        --prompt_file "data/prompts/prompt_${lang}.txt" \
        --save_invalid "$OUTPUT_DIR/invalid_samples_test_${lang}.json"
    
    echo "  ✓ Preprocessing complete for $lang"
done

echo ""
echo "Step 4: Generating dataset statistics..."
echo "=============================================="

python "$SCRIPTS_DIR/analyze_dataset_statistics.py" \
    --datasets \
        "$DATA_DIR/TISER_train.json" \
        "$OUTPUT_DIR/TISER_train_it_clean.json" \
        "$OUTPUT_DIR/TISER_train_fa_clean.json" \
        "$OUTPUT_DIR/TISER_train_de_clean.json" \
    --languages en it fa de \
    --output "$OUTPUT_DIR/dataset_statistics.json"

echo ""
echo "Step 5: Building mixed multilingual dataset..."
echo "=============================================="

python multilingual_tiser/preprocess/build_mixed_dataset.py \
    --en "$DATA_DIR/TISER_train.json" \
    --it "$OUTPUT_DIR/TISER_train_it_clean.json" \
    --fa "$OUTPUT_DIR/TISER_train_fa_clean.json" \
    --out "$OUTPUT_DIR/TISER_train_mixed.json" \
    --samples_per_lang 10000 \
    --seed 42

echo ""
echo "=============================================="
echo "✓ Pipeline complete!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - Translated datasets: $OUTPUT_DIR/TISER_*_{it,fa,de}.json"
echo "  - Cleaned datasets: $OUTPUT_DIR/TISER_*_{it,fa,de}_clean.json"
echo "  - Mixed dataset: $OUTPUT_DIR/TISER_train_mixed.json"
echo "  - Validation reports: $OUTPUT_DIR/validation_report_*.json"
echo "  - Statistics: $OUTPUT_DIR/dataset_statistics.json"
echo ""
