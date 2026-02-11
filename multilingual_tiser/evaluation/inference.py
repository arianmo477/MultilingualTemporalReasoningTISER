#!/usr/bin/env python3
import argparse
import json
import torch
import re
import string
import gc
from collections import Counter, defaultdict
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.io_gpu import balance_by_dataset_name
#from utils.evaluation import calculate_metrics
import random



# ==================================================
# METRIC UTILS (Optimized for Multi-Gold & Fixes)
# ==================================================
TAG_REGEX = re.compile(r"<[^>]+>")
ANSWER_REGEX = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_CUTOFF_REGEX = re.compile(r"<answer>\s*(.*)", re.DOTALL | re.IGNORECASE)

def normalize_text(text: str) -> str:
    if text is None: return ""
    text = str(text).lower().strip().translate(str.maketrans("", "", string.punctuation))
    # Standardize Persian characters
    text = text.replace("ي", "ی").replace("ك", "ک")
    return " ".join(text.split())

def normalize_boolean(text: str) -> str:
    t = normalize_text(text)
    trues = ["true", "yes", "vero", "ja", "درست", "wahr"]
    falses = ["false", "no", "falso", "nein", "نادرست", "falsch"]
    if any(x in t for x in trues): return "true"
    if any(x in t for x in falses): return "false"
    return ""

def calculate_metrics(pred: str, gold_candidates: list):
    """Checks prediction against multiple gold answers (e.g. Italian and English)."""
    def get_scores(p, g):
        if not g: return 0, 0, 0.0
        pb, gb = normalize_boolean(p), normalize_boolean(g)
        if gb:
            match = int(pb == gb)
            return match, match, float(match)
        
        p_norm = normalize_text(TAG_REGEX.sub("", str(p or "")))
        g_norm = normalize_text(TAG_REGEX.sub("", str(g or "")))
        
        em = int(p_norm == g_norm)
        soft = int(p_norm in g_norm or g_norm in p_norm)
        
        pt, gt = p_norm.split(), g_norm.split()
        if not pt or not gt: return em, soft, (1.0 if pt == gt else 0.0)
        common = Counter(pt) & Counter(gt)
        overlap = sum(common.values())
        f1 = 2 * overlap / (len(pt) + len(gt))
        return em, soft, f1

    best = (0, 0, 0.0)
    for gold in gold_candidates:
        res = get_scores(pred, gold)
        if res[0] > best[0] or (res[0] == best[0] and res[2] > best[2]):
            best = res
    return best

def extract_answer_from_generation(full_text: str) -> str:
    if not full_text: return ""
    m = ANSWER_REGEX.search(full_text)
    if m: return m.group(1).strip()
    m_cut = ANSWER_CUTOFF_REGEX.search(full_text)
    if m_cut: return re.split(r'\n\n|<', m_cut.group(1).strip())[0].strip()
    return full_text.strip()

# ================= PROMPT BUILDER =================
def build_prompt(ex):
    instr = ex.get("prompt", "")
    ctx = ex.get("temporal_context", ex.get("context", ""))
    qst = ex.get("question", "")
    # Template
    prompt = f"{instr}\n\nContext:\n{ctx}\n\nQuestion:\n{qst}\n\n"
    return prompt

# ================= GENERATION STRATEGY =================
def generate_with_strategy(model, tokenizer, batch_prompts, args):
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            
            repetition_penalty=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    input_len = inputs.input_ids.shape[1]
    decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

    if args.strategy == "base":
        return decoded

    # Iterative Continuation logic
    final_responses = decoded
    for i in range(len(final_responses)):
        ext_count = 0
        while "</answer>" not in final_responses[i] and ext_count < args.max_extensions:
            cont_prompt = batch_prompts[i] + final_responses[i]
            cont_inputs = tokenizer(cont_prompt, return_tensors="pt", truncation=True, max_length=3800).to(model.device)
            
            with torch.inference_mode():
                cont_out = model.generate(**cont_inputs, max_new_tokens=256, do_sample=False, repetition_penalty=1.1)
            
            new_text = tokenizer.decode(cont_out[0][cont_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            final_responses[i] += new_text
            ext_count += 1
            if "</answer>" in new_text: break
            
    return final_responses

# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    # NEW ARGUMENTS ADDED HERE
    parser.add_argument("--strategy", choices=["base", "iterative"], default="base")
    parser.add_argument("--max_extensions", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--only_passed", action="store_true", default=True)
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    # Load Tokenizer & Model
    print(f"Loading Model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    # Load Data
    dataset = load_dataset("json", data_files=args.test_file)["train"]
    if args.only_passed and "validation_status" in dataset.column_names:
        dataset = dataset.filter(lambda x: x["validation_status"] == "PASS")
    if args.max_eval_samples:
        dataset = balance_by_dataset_name(dataset.shuffle(seed=42), max_samples=args.max_eval_samples)

    print(f"Running strategy '{args.strategy}' on {len(dataset)} samples...")

    data_list = list(dataset)
    results = []
    r_em, r_f1, total = 0, 0, 0

    for i in tqdm(range(0, len(data_list), args.batch_size)):
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        
        batch_data = data_list[i : i + args.batch_size]
        batch_prompts = [build_prompt(ex) for ex in batch_data]

        decoded_preds = generate_with_strategy(model, tokenizer, batch_prompts, args)

        for j, pred in enumerate(decoded_preds):
            full_response = pred if pred.startswith("<reasoning>") else "<reasoning>" + pred
            gold_it = str(batch_data[j].get("answer", ""))
            gold_en = str(batch_data[j].get("answer_en", ""))
            
            extracted = extract_answer_from_generation(full_response)
            em, soft, f1 = calculate_metrics(extracted, [gold_it, gold_en])
            
            r_em += em
            r_f1 += f1
            total += 1
            
            results.append({
                "question_id": batch_data[j].get("question_id"),
                "gold_it": gold_it,
                "gold_en": gold_en,
                "extracted": extracted,
                "model_output": full_response,
                "em": em, "f1": f1
            })

            if total % args.log_every == 0:
                tqdm.write(f"Step {total} | EM: {r_em/total:.4f} | F1: {r_f1/total:.4f}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFINAL METRICS ({args.strategy}): EM: {r_em/total:.4f} | F1: {r_f1/total:.4f}")

if __name__ == "__main__":
    main()