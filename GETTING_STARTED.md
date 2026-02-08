# Getting Started with TISER Multilingual Extension

Quick guide to get you up and running with the multilingual TISER dataset extension.

## ⚡ Quick Start (5 minutes)

### 1. Setup Environment
```bash
# Clone and enter the repository
cd MultilingualTemporalReasoningTISER

# Create conda environment
conda env create -f environment.yml
conda activate tiser

# Make scripts executable
chmod +x scripts/*.sh
chmod +x multilingual_tiser/**/*.sh
```

### 2. Verify Everything Works
```bash
# Check your data is loaded correctly (should show 54,488 samples)
python scripts/analyze_dataset_statistics.py \
    --datasets data/TISER_train.json \
    --languages en \
    --output data/translated/statistics_en.json
```

**Expected output:**
```
Total Samples: 54488
Dataset Distribution:
  timeqa_split_hard_train    : 13534 (24.8%)
  tempreason_split_l2_train  : 13533 (24.8%)
  ...
```

### 3. Test Translation (Small Sample)
```bash
# Translate 100 samples to Italian (takes ~2-3 minutes on GPU)
python multilingual_tiser/translation/translate_dataset.py \
    --input data/TISER_train.json \
    --output_dir data/translated \
    --category train \
    --language it \
    --batch_size 8 \
    --max_samples 100
```

**What this does:**
- ✅ Loads NLLB translation model (~2GB)
- ✅ Translates 100 training samples to Italian
- ✅ Preserves entities in parentheses
- ✅ Maintains CoT structure
- ✅ Saves to `data/translated/TISER_train_it.json`

---

## 📊 Understanding Your Data

### Data Format (JSONL)
Your data files are in JSONL format (one JSON object per line):
```json
{"dataset_name": "tgqa_split_train", "question_id": "story42_Q1_0", ...}
{"dataset_name": "tgqa_split_train", "question_id": "story410_Q2_0", ...}
```

### Training Data Structure
Each training sample contains:
```json
{
  "dataset_name": "tgqa_split_train",
  "question_id": "story42_Q1_0",
  "question": "Which event started first...",
  "answer": "Event A",
  "prompt": "You are an AI assistant...",
  "output": "<reasoning>...</reasoning><answer>Event A</answer>"
}
```

### Test Data Structure
Test samples are similar but WITHOUT the `output` field.

---

## 🎯 Common Workflows

### Workflow 1: Quick Quality Check
**Goal:** Verify translation quality with a small sample

```bash
# 1. Translate 100 samples
python multilingual_tiser/translation/translate_dataset.py \
    --input data/TISER_train.json \
    --output_dir data/translated \
    --category train \
    --language it \
    --max_samples 100 \
    --batch_size 8

# 2. Validate quality
python scripts/validate_translation_quality.py \
    --original data/TISER_train.json \
    --translated data/translated/TISER_train_it.json \
    --language it \
    --sample_size 50 \
    --output validation_report.json

# 3. Review report
cat validation_report.json | python -m json.tool
```

### Workflow 2: Translate Single Language (Production)
**Goal:** Full translation for one language

```bash
# Translate all training data (~2-3 hours on GPU)
python multilingual_tiser/translation/translate_dataset.py \
    --input data/TISER_train.json \
    --output_dir data/translated \
    --category train \
    --language it \
    --batch_size 8 \
    --max_samples 0  # 0 = all samples

# Translate all test data
python multilingual_tiser/translation/translate_dataset.py \
    --input data/TISER_test.json \
    --output_dir data/translated \
    --category test \
    --language it \
    --batch_size 8 \
    --max_samples 0

# Clean and validate
python multilingual_tiser/preprocess/validate_tiser_dataset.py \
    --input data/translated/TISER_train_it.json \
    --output data/translated/TISER_train_it_clean.json \
    --split train \
    --prompt_file data/prompts/prompt_it.txt
```

### Workflow 3: All Languages (Full Pipeline)
**Goal:** Complete multilingual dataset creation

```bash
# Run everything (several hours)
./scripts/run_full_pipeline.sh

# Results will be in data/translated/
```

---

## 🔍 Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch size
```bash
--batch_size 4  # or even 2
```

### Issue: Translation Too Slow
**Cause:** Running on CPU instead of GPU

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions:**
- Ensure CUDA is installed
- Check `nvidia-smi` for GPU status
- Use smaller model if GPU memory is limited

### Issue: "FileNotFoundError"
**Cause:** Translated files don't exist yet

**Solution:** Run translation first:
```bash
./scripts/example_translate_single_language.sh
```

### Issue: Bad Translation Quality
**Possible causes:**
1. Entity masking not working (check validation report)
2. Model too small (try larger NLLB model)
3. Source data issues (check original dataset)

**Debug:**
```bash
# Check validation report
python scripts/validate_translation_quality.py \
    --original data/TISER_train.json \
    --translated data/translated/TISER_train_it.json \
    --language it \
    --sample_size 200 \
    --export_samples problematic.json

# Review problematic samples
cat problematic.json | python -m json.tool | less
```

---

## 📈 Next Steps

### After Translation
1. **Validate Quality** - Check reports, review samples
2. **Clean Data** - Use preprocessing scripts
3. **Generate Statistics** - Compare languages
4. **Manual Review** - Sample 50-100 translations
5. **Train Models** - Use cleaned data with your training pipeline

### Integration with Training
```python
from utils.io_gpu import load_json

# Load your cleaned data
train_it = load_json("data/translated/TISER_train_it_clean.json")

# Each sample has all you need
for sample in train_it:
    question = sample['question']      # Translated question
    answer = sample['answer']          # Translated answer
    output = sample['output']          # Translated CoT reasoning
    prompt = sample['prompt']          # Language-specific prompt
```

---

## 📚 More Resources

- **[USAGE.md](USAGE.md)** - Complete reference guide
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - What was improved
- **[README.md](README.md)** - Project overview

---

## 💡 Tips

1. **Start Small**: Always test with 100-1000 samples first
2. **Monitor GPU**: Use `watch -n 1 nvidia-smi` during translation
3. **Save Progress**: Translation is incremental, safe to stop/resume
4. **Validate Often**: Run validation after each translation
5. **Manual Review**: Critical for ensuring quality

---

## ✅ Success Checklist

Before using translated data for training:

- [ ] Translation completed without errors
- [ ] Validation report shows <10% issue rate
- [ ] Entity preservation >95%
- [ ] Manual review of 100 samples looks good
- [ ] Statistics match expected distributions
- [ ] Cleaned data passes all checks

---

**Ready to start? Run:**
```bash
./scripts/example_translate_single_language.sh
```

This will translate 1000 samples to Italian in ~5-10 minutes, giving you a feel for the process!
