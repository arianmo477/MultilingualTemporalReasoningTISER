#!/usr/bin/env python3
"""
Advanced Validation Script:
1. Scores translation quality (Answer vs English Answer).
2. IF TRAIN: Validates and Fixes CoT structure (Loops, Missing Tags).
3. IF TEST: Skips structure check (as output field doesn't exist).
"""

import json
import argparse
import re
import os
from typing import Dict, List, Tuple
from collections import Counter
from sentence_transformers import SentenceTransformer, util

# ==================================================
# CONFIGURATION
# ==================================================
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SCORE_PASS = 0.85      
SCORE_THRESHOLD = 0.50 

# ==================================================
# STRUCTURAL VALIDATION TOOLS
# ==================================================

ALLOWED_TAGS = {
    "reasoning", "/reasoning",
    "timeline", "/timeline",
    "reflection", "/reflection",
    "answer", "/answer",
}

TAG_FINDER = re.compile(r"<\s*/?\s*([a-zA-Z0-9_]+)[^>]*>")

def has_only_allowed_tags(text: str) -> bool:
    """
    Returns False if ANY tag other than the allowed CoT tags appears.
    """
    if not text:
        return False

    tags = TAG_FINDER.findall(text)

    for tag in tags:
        tag = tag.strip().lower()
        if tag not in ALLOWED_TAGS:
            return False

    return True



def validate_cot_structure(text: str) -> bool:
    """Quick check if CoT structure is valid"""
    if not text: return False
    required = ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]
    if not all(tag in text for tag in required):
        return False
    return True

def clean_repetitions_aggressive(text: str) -> str:
    """Nuclear cleanup for loops"""
    text = re.sub(r'(?i)(\b[a-z]\s+){5,}', ' ', text)
    text = re.sub(r'(.)\1{5,}', r'\1', text)
    text = re.sub(r'(?i)(tg\s*){3,}', ' ', text)
    text = re.sub(r'(\b\w+\b)(\s*,\s*\1){3,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def repair_cot_structure(cot_text: str, final_answer: str) -> str:
    """Rebuilds the XML structure if broken"""
    cot_text = clean_repetitions_aggressive(cot_text)
    
    if "<reasoning>" not in cot_text:
        cot_text = "<reasoning>\n" + cot_text

    if "<timeline>" not in cot_text and "</timeline>" not in cot_text:
         cot_text += "\n<timeline>\n(Implied)\n</timeline>"
    if "<reflection>" not in cot_text and "</reflection>" not in cot_text:
         cot_text += "\n<reflection>\n(None)\n</reflection>"

    if "</reasoning>" not in cot_text:
        if "</reflection>" in cot_text:
            cot_text = cot_text.replace("</reflection>", "</reflection>\n</reasoning>")
        else:
            cot_text += "\n</reasoning>"

    cot_text = re.sub(r"<answer>.*", "", cot_text, flags=re.DOTALL)
    cot_text = re.sub(r"</(timeline|reflection|reasoning)>[_\W]*$", "", cot_text.strip())
    
    if "<reasoning>" in cot_text and "</reasoning>" not in cot_text:
         cot_text += "\n</reasoning>"

    cot_text = cot_text.strip() + f"\n\n<answer>\n{final_answer}\n</answer>"
    return cot_text

# ==================================================
# MAIN LOGIC
# ==================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Translated file (Target Language)")
    parser.add_argument("--en", required=True, help="English source file (Ground Truth)")
    parser.add_argument("--output", required=True, help="Where to save the fixed file")
    parser.add_argument("--category", required=True, choices=["train", "test"], help="Dataset category")
    parser.add_argument("--lang", default="it", help="Target language code")
    args = parser.parse_args()

    print(f"Loading scoring model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading data...")
    with open(args.input, 'r', encoding='utf-8') as f:
        tgt_data = json.load(f)
    
    with open(args.en, 'r', encoding='utf-8') as f:
        en_data = json.load(f)
        en_map = {x['question_id']: x for x in en_data}

    stats = Counter()
    fixed_data = []

    print(f"Validating and Scoring (Mode: {args.category.upper()})...")
    
    for item in tgt_data:
        qid = item.get("question_id")
        tgt_answer = item.get("answer", "").strip()
        
        # 1. Get Ground Truth
        en_item = en_map.get(qid)
        if not en_item:
            item["validation_status"] = "FAIL"
            item["validation_reason"] = "missing_english_pair"
            fixed_data.append(item)
            stats["missing_en"] += 1
            continue

        en_answer = en_item.get("answer", "").strip()
        item["answer_en"] = en_answer 

        # 2. Structural Fix (ONLY FOR TRAIN)
        was_fixed = False
        if args.category == "train":
            tgt_output = item.get("output", "")
            #  STRICT TAG VALIDATION
            if not has_only_allowed_tags(tgt_output):
                item["validation_status"] = "FAIL"
                item["validation_reason"] = "invalid_extra_tags"
                stats["FAIL"] += 1
                fixed_data.append(item)
                continue
            # Only validate structure if output field exists
            if not validate_cot_structure(tgt_output):
                new_output = repair_cot_structure(tgt_output, tgt_answer)
                if new_output != tgt_output:
                    item["output"] = new_output
                    was_fixed = True
                    stats["structure_fixed"] += 1
        
        # For TEST mode, we ignore 'output' field entirely

        # 3. Semantic Scoring (Always check Answer quality)
        if tgt_answer and en_answer:
            embeddings = model.encode([tgt_answer, en_answer], convert_to_tensor=True)
            score = util.cos_sim(embeddings[0], embeddings[1]).item()
        else:
            score = 0.0
        
        item["translation_score"] = float(score)

        # 4. Final Status Determination
        status = "FAIL"
        reason = "unknown"

        if score < SCORE_THRESHOLD:
            status = "FAIL"
            reason = "low_semantic_score"
        elif score >= SCORE_PASS:
            if was_fixed:
                status = "FIXED" 
                reason = "high_score_structure_repaired"
            else:
                status = "PASS"
                if score >= 1.0:
                    reason = "perfect_match"
                else:
                    reason = "high_semantic_score"
        else:
            if was_fixed:
                status = "FIXED"
                reason = "medium_score_structure_repaired"
            else:
                status = "AMBIGUOUS" 
                reason = "medium_semantic_score"

        item["validation_status"] = status
        item["validation_reason"] = reason
        
        stats[status] += 1
        fixed_data.append(item)

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print(f"RESULTS SUMMARY ({args.category.upper()})")
    print("="*40)
    print(f"Total Processed: {len(fixed_data)}")
    print(f"PASS           : {stats['PASS']}")
    print(f"FIXED          : {stats['FIXED']}")
    print(f"AMBIGUOUS      : {stats['AMBIGUOUS']}")
    print(f"FAIL           : {stats['FAIL']}")
    print(f"Missing EN     : {stats['missing_en']}")
    print("-" * 40)
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()