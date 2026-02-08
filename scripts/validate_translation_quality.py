#!/usr/bin/env python3
"""
Translation Quality Validation Script

This script samples translations and provides metrics to assess quality.
Helps identify potential issues in the translation pipeline.
"""

import argparse
import random
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io_gpu import load_json, save_json


def check_entity_preservation(original, translated, entities):
    """Check if entities were properly preserved"""
    issues = []
    for entity in entities:
        if entity in original and entity not in translated:
            issues.append(f"Missing entity: {entity}")
    return issues

def check_length_ratio(original, translated, min_ratio=0.5, max_ratio=2.0):
    """Check if translation length is reasonable"""
    if not original or not translated:
        return []
    
    ratio = len(translated) / len(original)
    if ratio < min_ratio:
        return [f"Translation too short (ratio: {ratio:.2f})"]
    elif ratio > max_ratio:
        return [f"Translation too long (ratio: {ratio:.2f})"]
    return []

def check_cot_structure(text):
    """Check if CoT structure is preserved"""
    required_tags = ['<reasoning>', '</reasoning>', '<timeline>', '</timeline>', 
                     '<reflection>', '</reflection>', '<answer>', '</answer>']
    issues = []
    for tag in required_tags:
        if tag not in text:
            issues.append(f"Missing tag: {tag}")
    return issues

def extract_entities(text):
    """Extract entities in parentheses"""
    import re
    pattern = re.compile(r'\(([A-Z][^()]+)\)')
    return pattern.findall(text)

def validate_sample(original_sample, translated_sample, language):
    """Validate a single translation"""
    issues = defaultdict(list)
    
    # Check question translation
    orig_q = original_sample.get('question', '')
    trans_q = translated_sample.get('question', '')
    
    entities = extract_entities(orig_q)
    issues['question_entities'] = check_entity_preservation(orig_q, trans_q, entities)
    issues['question_length'] = check_length_ratio(orig_q, trans_q)
    
    # Check answer translation
    orig_a = original_sample.get('answer', '')
    trans_a = translated_sample.get('answer', '')
    issues['answer_length'] = check_length_ratio(orig_a, trans_a, 0.3, 3.0)
    
    # Check CoT structure (training data only)
    if 'output' in translated_sample:
        issues['cot_structure'] = check_cot_structure(translated_sample['output'])
    
    # Check language tag
    if translated_sample.get('language') != language:
        issues['language_tag'] = [f"Incorrect language tag: {translated_sample.get('language')}"]
    
    return {k: v for k, v in issues.items() if v}

def main():
    parser = argparse.ArgumentParser(description="Validate translation quality")
    parser.add_argument("--original", required=True, help="Path to original English dataset")
    parser.add_argument("--translated", required=True, help="Path to translated dataset")
    parser.add_argument("--language", required=True, choices=['it', 'fa', 'de'], help="Target language")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to validate")
    parser.add_argument("--output", default="validation_report.json", help="Output report path")
    parser.add_argument("--export_samples", help="Export problematic samples to this file")
    args = parser.parse_args()
    
    print(f"Loading datasets...")
    original_data = load_json(args.original)
    translated_data = load_json(args.translated)
    
    print(f"Original dataset size: {len(original_data)}")
    print(f"Translated dataset size: {len(translated_data)}")
    
    # Create mapping by question_id
    orig_map = {item['question_id']: item for item in original_data}
    trans_map = {item['question_id']: item for item in translated_data}
    
    # Find common IDs
    common_ids = list(set(orig_map.keys()) & set(trans_map.keys()))
    print(f"Common question IDs: {len(common_ids)}")
    
    # Sample for validation
    sample_ids = random.sample(common_ids, min(args.sample_size, len(common_ids)))
    
    print(f"\nValidating {len(sample_ids)} samples...")
    
    all_issues = []
    issue_counts = defaultdict(int)
    
    for qid in sample_ids:
        issues = validate_sample(orig_map[qid], trans_map[qid], args.language)
        if issues:
            all_issues.append({
                'question_id': qid,
                'issues': issues,
                'original_question': orig_map[qid]['question'][:100] + '...',
                'translated_question': trans_map[qid]['question'][:100] + '...'
            })
            for issue_type in issues:
                issue_counts[issue_type] += 1
    
    # Generate report
    report = {
        'language': args.language,
        'total_samples': len(sample_ids),
        'samples_with_issues': len(all_issues),
        'issue_rate': len(all_issues) / len(sample_ids),
        'issue_counts': dict(issue_counts),
        'problematic_samples': all_issues[:20]  # Top 20
    }
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"Language: {args.language}")
    print(f"Samples validated: {len(sample_ids)}")
    print(f"Samples with issues: {len(all_issues)} ({report['issue_rate']*100:.1f}%)")
    print(f"\nIssue Breakdown:")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  - {issue_type}: {count}")
    
    # Save report
    save_json(args.output, report)
    print(f"\nFull report saved to: {args.output}")
    
    # Export problematic samples if requested
    if args.export_samples:
        export_data = [
            {
                'original': orig_map[item['question_id']],
                'translated': trans_map[item['question_id']],
                'issues': item['issues']
            }
            for item in all_issues
        ]
        save_json(args.export_samples, export_data)
        print(f"Problematic samples exported to: {args.export_samples}")
    
    print("\n✓ Validation complete!")

if __name__ == "__main__":
    main()
