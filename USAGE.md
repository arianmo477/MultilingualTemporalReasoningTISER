# TISER Multilingual Extension - Usage Guide

This guide explains how to use the multilingual extension for the TISER (Timeline Self-Reflection) temporal reasoning dataset.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Translation Pipeline](#translation-pipeline)
- [Validation and Quality Control](#validation-and-quality-control)
- [Data Preprocessing](#data-preprocessing)
- [Dataset Statistics](#dataset-statistics)
- [Full Pipeline](#full-pipeline)

---

## Installation

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate tiser
```

### 2. Make Scripts Executable
```bash
chmod +x scripts/*.sh
chmod +x multilingual_tiser/**/*.sh
```

### 3. Verify Installation
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Quick Start

### Translate to a Single Language (Small Sample)
```bash
# Translate 1000 training samples to Italian
python multilingual_tiser/translation/translate_dataset.py \
    --input data/TISER_train.json \
    --output_dir data/translated \
    --category train \
    --language it \
    --batch_size 8 \
    --max_samples 1000
```

**Output:** `data/translated/TISER_train_it.json`

### Use Example Scripts
```bash
# Quick translation example
./scripts/example_translate_single_language.sh

# Validate and clean
./scripts/example_validate_and_clean.sh

# Analyze statistics
./scripts/example_analyze_statistics.sh
```

---

## Translation Pipeline

### Supported Languages
- **Italian (it)**: `ita_Latn`
- **Persian/Farsi (fa)**: `pes_Arab`
- **German (de)**: `deu_Latn`

### Basic Translation Command
```bash
python multilingual_tiser/translation/translate_dataset.py \
    --input <input_file> \
    --output_dir <output_directory> \
    --category {train|test} \
    --language {it|fa|de} \
    --batch_size 8 \
    --max_samples 0  # 0 = all samples
```

### Parameters
- `--input`: Path to original English TISER dataset (JSON)
- `--output_dir`: Directory to save translated files
- `--category`: Dataset type (`train` or `test`)
- `--language`: Target language code
- `--batch_size`: Batch size for translation (default: 8)
- `--max_samples`: Maximum samples to translate (0 = all)

### Translation Features
1. **Entity Masking**: Preserves proper names in parentheses
2. **Chunking**: Handles long texts without truncation
3. **CoT Structure Preservation**: Maintains `<reasoning>`, `<timeline>`, `<reflection>`, `<answer>` tags
4. **Language-Specific Normalization**: 
   - Persian digit conversion
   - Boolean normalization
   - Temporal expression handling

### Example: Translate All Languages
```bash
for lang in it fa de; do
    python multilingual_tiser/translation/translate_dataset.py \
        --input data/TISER_train.json \
        --output_dir data/translated \
        --category train \
        --language $lang \
        --batch_size 8 \
        --max_samples 0
done
```

---

## Validation and Quality Control

### Check Translation Quality
```bash
python scripts/validate_translation_quality.py \
    --original data/TISER_train.json \
    --translated data/translated/TISER_train_it.json \
    --language it \
    --sample_size 200 \
    --output validation_report.json \
    --export_samples problematic_samples.json
```

### What Gets Validated
- ✅ Entity preservation
- ✅ Length ratios (translation shouldn't be too short/long)
- ✅ CoT structure integrity
- ✅ Language tag consistency

### Output Files
- **validation_report.json**: Summary statistics and issue counts
- **problematic_samples.json**: Samples with detected issues for manual review

---

## Data Preprocessing

### Clean and Normalize Dataset
```bash
python multilingual_tiser/preprocess/validate_tiser_dataset.py \
    --input data/translated/TISER_train_it.json \
    --output data/translated/TISER_train_it_clean.json \
    --split train \
    --prompt_file data/prompts/prompt_it.txt \
    --save_invalid invalid_samples.json
```

### Preprocessing Steps
1. **Unicode Repair**: Fixes mangled characters
2. **Deduplication**: Removes duplicate question IDs
3. **Unknown Answer Filtering**: Removes samples with "unknown" answers
4. **Structural Validation**: Ensures all required fields are present
5. **Prompt Canonicalization**: Uses consistent prompts from file

### Build Mixed Multilingual Dataset
```bash
python multilingual_tiser/preprocess/build_mixed_dataset.py \
    --en data/TISER_train.json \
    --it data/translated/TISER_train_it_clean.json \
    --fa data/translated/TISER_train_fa_clean.json \
    --out data/TISER_train_mixed.json \
    --samples_per_lang 10000 \
    --seed 42
```

---

## Dataset Statistics

### Analyze Single or Multiple Datasets
```bash
python scripts/analyze_dataset_statistics.py \
    --datasets \
        data/TISER_train.json \
        data/translated/TISER_train_it_clean.json \
        data/translated/TISER_train_fa_clean.json \
    --languages en it fa \
    --output statistics.json
```

### Generated Statistics
- Total sample counts
- Dataset distribution (TGQA, TempReason, TimeQA, etc.)
- Question type distribution
- Average text lengths
- CoT structure coverage
- Cross-language comparisons

---

## Full Pipeline

### Run Complete Translation and Validation Pipeline
```bash
./scripts/run_full_pipeline.sh
```

### Pipeline Steps
1. **Translation**: Translates train/test data to all languages
2. **Validation**: Checks translation quality
3. **Preprocessing**: Cleans and normalizes datasets
4. **Statistics**: Generates comprehensive reports
5. **Mixed Dataset**: Creates multilingual training set

### Expected Output Structure
```
data/translated/
├── TISER_train_it.json              # Raw Italian translation
├── TISER_train_it_clean.json        # Cleaned version
├── TISER_test_it.json
├── TISER_test_it_clean.json
├── TISER_train_fa.json              # Raw Persian translation
├── TISER_train_fa_clean.json
├── TISER_test_fa.json
├── TISER_test_fa_clean.json
├── TISER_train_de.json              # Raw German translation
├── TISER_train_de_clean.json
├── TISER_test_de.json
├── TISER_test_de_clean.json
├── TISER_train_mixed.json           # Mixed multilingual dataset
├── validation_report_*.json         # Quality reports
├── invalid_samples_*.json           # Filtered samples
└── dataset_statistics.json          # Cross-language statistics
```

---

## Advanced Usage

### Custom Translation Models

To use a different translation model, modify [`translate_dataset.py`](multilingual_tiser/translation/translate_dataset.py):
```python
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"  # Larger model
# or
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
```

### Language-Specific Temporal Normalization

The pipeline now supports language-specific temporal expression normalization:
- **English**: "starts at 1990" → "started in 1990"
- **Italian**: "dal 1990 al 1995" → "Dal 1990 al 1995"
- **German**: "von 1990 bis 1995" → "Von 1990 bis 1995"
- **Persian**: "از ۱۹۹۰ تا ۱۹۹۵" → "از ۱۹۹۰ تا ۱۹۹۵"

### Batch Processing with GPU

Adjust batch size based on your GPU memory:
- **8GB GPU**: `--batch_size 4`
- **16GB GPU**: `--batch_size 8`
- **24GB+ GPU**: `--batch_size 16`

---

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
--batch_size 4

# Or use CPU (slower)
CUDA_VISIBLE_DEVICES="" python translate_dataset.py ...
```

### Translation Quality Issues
1. Check validation report for specific issues
2. Review problematic samples manually
3. Adjust entity masking if needed
4. Consider using larger translation model

### Unicode Issues (Persian)
The pipeline automatically handles Persian digits and RTL text. If you encounter issues:
```python
from utils.evaluation import normalize_persian_digits
text = normalize_persian_digits(text)
```

---

## Tips for Best Results

1. **Start Small**: Test with `--max_samples 100` first
2. **Validate Often**: Run validation after each translation
3. **Manual Review**: Sample 50-100 translations manually
4. **Mixed Training**: Use mixed datasets for better cross-lingual performance
5. **Monitor GPU**: Use `nvidia-smi` to track memory usage

---

## Citation

If you use this multilingual extension, please cite the original TISER paper:
```bibtex
@misc{bazaga2025learningreasontimetimeline,
      title={Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models}, 
      author={Adrián Bazaga and Rexhina Blloshmi and Bill Byrne and Adrià de Gispert},
      year={2025},
      eprint={2504.05258},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.05258}, 
}
```

---

## Support

For issues or questions:
1. Check the validation reports
2. Review the problematic samples
3. Ensure all dependencies are installed
4. Check GPU availability for translation

For more details, see the main [README.md](README.md).
