#!/usr/bin/env python3
import argparse
import json
import re
import string
import gc
import torch
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from utils.utils import extract_answer_from_generation, calculate_metrics

# ============================================================
# PROMPTING
# ============================================================

def build_prompt(instr: str, ctx: str, qst: str, lang: str) -> str:
    instr = instr or ""
    ctx = ctx or ""
    qst = qst or ""
    lang = lang.lower() if lang else "en"

    # Default Rules (English)
    rules_text = (
        "\n\nYou MUST follow these rules:\n"
        "1) Output ONLY two blocks in this exact order:\n"
        "   <reasoning>...</reasoning>\n"
        "   <answer>...</answer>\n"
        "2) The <answer> content MUST be in ENGLISH ONLY.\n" 
        "3) The <answer> MUST be a single short phrase. No full sentences.\n"
        "4) Do NOT repeat the question.\n"
        "5) If you cannot find the answer from the context, output exactly: Unknown\n"
        "6) If the question is yes/no, output exactly: True or False\n"
        "7) Do not add any text outside the two tags.\n"
    )

    if lang == 'it': # ITALIAN
        rules_text = (
            "\n\nDEVI seguire queste regole:\n"
            "1) Fornisci SOLO due blocchi in questo esatto ordine:\n"
            "   <reasoning>...</reasoning>\n"
            "   <answer>...</answer>\n"
            "2) Il contenuto di <answer> DEVE essere in ITALIANO.\n"
            "3) La risposta deve essere una breve frase singola. Niente frasi complete.\n"
            "4) NON ripetere la domanda.\n"
            "5) Se non trovi la risposta nel contesto, scrivi esattamente: Sconosciuto\n"
            "6) Se la domanda è sì/no, scrivi esattamente: Vero o Falso\n"
            "7) Non aggiungere altro testo fuori dai tag.\n"
        )
    elif lang == 'fa': # PERSIAN
        rules_text = (
            "\n\nشما باید این قوانین را رعایت کنید:\n"
            "1) فقط دو بلوک را دقیقاً به این ترتیب خروجی دهید:\n"
            "   <reasoning>...</reasoning>\n"
            "   <answer>...</answer>\n"
            "2) محتوای <answer> باید فقط به زبان فارسی باشد.\n"
            "3) پاسخ باید یک عبارت کوتاه باشد. جمله کامل ننویسید.\n"
            "4) سوال را تکرار نکنید.\n"
            "5) اگر پاسخ را در متن پیدا نکردید، دقیقاً بنویسید: نامشخص\n"
            "6) اگر سوال بله/خیر است، بنویسید: درست یا نادرست\n"
            "7) هیچ متنی خارج از این دو تگ اضافه نکنید.\n"
        )
    elif lang == 'de':  # GERMAN
        rules_text = (
        "\n\nDu MUSST diese Regeln befolgen:\n"
        "1) Gib NUR zwei Blöcke in genau dieser Reihenfolge aus:\n"
        "   <reasoning>...</reasoning>\n"
        "   <answer>...</answer>\n"
        "2) Der Inhalt von <answer> MUSS auf DEUTSCH sein.\n"
        "3) Die Antwort muss eine kurze einzelne Phrase sein. Keine vollständigen Sätze.\n"
        "4) Wiederhole die Frage NICHT.\n"
        "5) Wenn die Antwort nicht im Kontext steht, schreibe exakt: Unbekannt\n"
        "6) Bei Ja/Nein-Fragen schreibe exakt: Wahr oder Falsch\n"
        "7) Füge keinen Text außerhalb der Tags hinzu.\n"
    )
    
    user_content = (
        f"{instr.strip()}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question:\n{qst}\n"
    )
    return user_content


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--only_passed",
        action="store_true",
        help="Evaluate only samples with validation_status == PASS"
    )

    args = parser.parse_args()

    # Load Tokenizer
    print(f"Loading Base Model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quant
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading Adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    dataset = load_dataset("json", data_files=args.test_file)["train"]

    if args.only_passed:
        if "validation_status" not in dataset.column_names:
            print("Warning: validation_status column not found, skipping filtering.")
        else:
            before = len(dataset)
            dataset = dataset.filter(lambda x: x["validation_status"] == "PASS")
            after = len(dataset)
            print(f"Filtered evaluation set: {before} → {after} (PASS only)")

    if args.max_eval_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_eval_samples)))

    preds = []
    total = 0
    sum_em = 0
    sum_soft_em = 0
    sum_f1 = 0.0
    skipped_oom = 0

    print(f"Starting Evaluation on {len(dataset)} samples...")

    for i, ex in enumerate(tqdm(dataset)):
        # Proactive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        instr = ex.get("prompt", "")
        ctx = ex.get("temporal_context", ex.get("context", ""))
        qst = ex.get("question", "")
        gold_answer = str(ex.get("answer", ""))
        
        lang = ex.get("language", "en")

        user_content = build_prompt(instr, ctx, qst, lang)

        if args.use_chat_template:
            messages = [{"role": "user", "content": user_content}]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ) + "<reasoning>"
        else:
            prompt_str = f"{user_content}\n\n<reasoning>"

        # Enforce truncation
        inputs = tokenizer(
            prompt_str, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1536
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 0.0),
            temperature=max(args.temperature, 1e-6) if args.temperature > 0.0 else None,
            top_p=args.top_p if args.temperature > 0.0 else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        try:
            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)

            # Decode only newly generated tokens
            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            full_response = "<reasoning>" + generated
            pred_extracted = extract_answer_from_generation(full_response)
            em, soft_em, f1 = calculate_metrics(pred_extracted, gold_answer)

            sum_em += em
            sum_soft_em += soft_em
            sum_f1 += f1
            total += 1

            preds.append({
                "question": qst,
                "gold": gold_answer,
                "pred_extracted": pred_extracted,
                "raw_output": full_response,
                "em": em,
                "soft_em": soft_em,
                "f1": f1,
            })

            # === NEW: Save Logic Inside Loop ===
            if total % args.log_every == 0:
                cur_em = sum_em / total
                cur_soft = sum_soft_em / total
                cur_f1 = sum_f1 / total
                
                tqdm.write(
                    f"Step {total} | EM: {cur_em:.3f} | Soft: {cur_soft:.3f} | F1: {cur_f1:.3f}"
                )
                
                # Save intermediate results
                intermediate_results = {
                    "Strict_EM": cur_em,
                    "Soft_Match": cur_soft,
                    "F1": cur_f1,
                    "total_samples": total,
                    "skipped_oom": skipped_oom,
                    "details": preds,
                }
                
                try:
                    with open(args.output_file, "w", encoding="utf-8") as f:
                        json.dump(intermediate_results, f, ensure_ascii=False, indent=2)
                except Exception as save_err:
                    tqdm.write(f"Warning: Failed to save intermediate results: {save_err}")
            # ===================================

        except torch.cuda.OutOfMemoryError:
            skipped_oom += 1
            tqdm.write(f"WARNING: OOM detected at sample {i}. Skipping this sample.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            continue
        except Exception as e:
            tqdm.write(f"ERROR at sample {i}: {str(e)}")
            continue

    if total == 0:
        print("\n===== NO SAMPLES EVALUATED SUCCESSFULLY =====")
        return

    final_results = {
        "Strict_EM": (sum_em / total),
        "Soft_Match": (sum_soft_em / total),
        "F1": (sum_f1 / total),
        "total_samples": total,
        "skipped_oom": skipped_oom,
        "details": preds,
    }

    print("\n===== FINAL RESULTS =====")
    print(f"Total Evaluated       : {total}")
    print(f"Skipped due to OOM    : {skipped_oom}")
    print(f"Strict EM (Exact Match) : {final_results['Strict_EM']:.4f}")
    print(f"Soft Match              : {final_results['Soft_Match']:.4f}")
    print(f"F1 Score                : {final_results['F1']:.4f}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()