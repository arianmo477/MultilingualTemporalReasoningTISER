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
from utils.evaluation import extract_answer_from_generation, TAG_REGEX, normalize_text, normalize_boolean, calculate_metrics, calculate_metrics_mix
#from utils.evaluation import calculate_metrics
import random




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

            # Extract basic fields
            language = batch_data[j].get("language", "en") # Default to en if missing
            if language == "en":
                gold_en = batch_data[j].get("answer", "").strip()
            else: 
                gold_en = batch_data[j].get("answer_en", "").strip()
                gold_tr = batch_data[j].get("answer", "").strip()
            
            extracted = extract_answer_from_generation(full_response)
            
            # --- CONDITIONAL METRIC CALCULATION ---
            if language == "en":
                # For English, we only compare against the main answer
                em, soft, f1 = calculate_metrics(extracted, gold_en)
            else:
                # For non-English, we mix metrics against target language answer AND English answer
                em, soft, f1 = calculate_metrics_mix(extracted, [gold_tr, gold_en])
            # --------------------------------------
           
            
            r_em += em
            r_f1 += f1
            total += 1
            
            results.append({
                "question_id": batch_data[j].get("question_id"),
                "gold_tr": gold_tr if language != "en" else None,
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