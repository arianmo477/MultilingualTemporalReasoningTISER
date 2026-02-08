#!/usr/bin/env python3
"""
Dataset Statistics and Analysis Script

Generates comprehensive statistics about the TISER datasets
to help understand data distribution and quality.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io_gpu import load_json

def analyze_dataset(data, language="en"):
    """Analyze a single dataset"""
    stats = {
        'total_samples': len(data),
        'language': language,
        'dataset_distribution': Counter(),
        'question_types': Counter(),
        'answer_lengths': [],
        'question_lengths': [],
        'has_output': 0,
        'has_temporal_context': 0,
    }
    
    for item in data:
        # Dataset distribution
        stats['dataset_distribution'][item.get('dataset_name', 'unknown')] += 1
        
        # Question types (heuristic)
        question = item.get('question', '').lower()
        if 'which event' in question or 'what event' in question:
            stats['question_types']['event_identification'] += 1
        elif 'how long' in question or 'duration' in question:
            stats['question_types']['duration'] += 1
        elif 'when did' in question or 'what year' in question:
            stats['question_types']['temporal_point'] += 1
        elif 'true or false' in question:
            stats['question_types']['boolean'] += 1
        elif 'how much time' in question:
            stats['question_types']['time_difference'] += 1
        else:
            stats['question_types']['other'] += 1
        
        # Lengths
        stats['answer_lengths'].append(len(item.get('answer', '')))
        stats['question_lengths'].append(len(item.get('question', '')))
        
        # Structural features
        if 'output' in item:
            stats['has_output'] += 1
        if item.get('temporal_context'):
            stats['has_temporal_context'] += 1
    
    # Calculate statistics
    stats['avg_answer_length'] = sum(stats['answer_lengths']) / len(stats['answer_lengths']) if stats['answer_lengths'] else 0
    stats['avg_question_length'] = sum(stats['question_lengths']) / len(stats['question_lengths']) if stats['question_lengths'] else 0
    stats['output_percentage'] = (stats['has_output'] / stats['total_samples']) * 100 if stats['total_samples'] else 0
    stats['temporal_context_percentage'] = (stats['has_temporal_context'] / stats['total_samples']) * 100 if stats['total_samples'] else 0
    
    return stats

def print_stats(stats, title):
    """Pretty print statistics"""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Language: {stats['language']}")
    print(f"\nDataset Distribution:")
    for dataset, count in stats['dataset_distribution'].most_common():
        percentage = (count / stats['total_samples']) * 100
        print(f"  {dataset:40s}: {count:6d} ({percentage:5.1f}%)")
    
    print(f"\nQuestion Types:")
    for qtype, count in stats['question_types'].most_common():
        percentage = (count / stats['total_samples']) * 100
        print(f"  {qtype:40s}: {count:6d} ({percentage:5.1f}%)")
    
    print(f"\nText Statistics:")
    print(f"  Avg Question Length: {stats['avg_question_length']:.1f} chars")
    print(f"  Avg Answer Length: {stats['avg_answer_length']:.1f} chars")
    print(f"  Samples with Output: {stats['output_percentage']:.1f}%")
    print(f"  Samples with Temporal Context: {stats['temporal_context_percentage']:.1f}%")

def compare_languages(stats_dict):
    """Compare statistics across languages"""
    print("\n" + "="*70)
    print("CROSS-LANGUAGE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<40s} " + " ".join([f"{lang:>10s}" for lang in stats_dict.keys()]))
    print("-" * 70)
    
    # Sample counts
    row = f"{'Total Samples':<40s} "
    for stats in stats_dict.values():
        row += f"{stats['total_samples']:>10d} "
    print(row)
    
    # Average lengths
    row = f"{'Avg Question Length':<40s} "
    for stats in stats_dict.values():
        row += f"{stats['avg_question_length']:>10.1f} "
    print(row)
    
    row = f"{'Avg Answer Length':<40s} "
    for stats in stats_dict.values():
        row += f"{stats['avg_answer_length']:>10.1f} "
    print(row)
    
    # Question type distribution
    print(f"\n{'Question Type Distribution (%)':}")
    all_qtypes = set()
    for stats in stats_dict.values():
        all_qtypes.update(stats['question_types'].keys())
    
    for qtype in sorted(all_qtypes):
        row = f"  {qtype:<38s} "
        for stats in stats_dict.values():
            count = stats['question_types'].get(qtype, 0)
            pct = (count / stats['total_samples']) * 100 if stats['total_samples'] else 0
            row += f"{pct:>9.1f}% "
        print(row)

def main():
    parser = argparse.ArgumentParser(description="Analyze TISER dataset statistics")
    parser.add_argument("--datasets", nargs='+', required=True, 
                       help="Paths to dataset JSON files")
    parser.add_argument("--languages", nargs='+', required=True,
                       help="Language codes corresponding to datasets")
    parser.add_argument("--output", default="dataset_statistics.json",
                       help="Output JSON file for statistics")
    args = parser.parse_args()
    
    if len(args.datasets) != len(args.languages):
        raise ValueError("Number of datasets must match number of languages")
    
    all_stats = {}
    
    for dataset_path, language in zip(args.datasets, args.languages):
        print(f"\nLoading {language} dataset from {dataset_path}...")
        data = load_json(dataset_path)
        stats = analyze_dataset(data, language)
        all_stats[language] = stats
        print_stats(stats, f"Statistics for {language.upper()} Dataset")
    
    # Cross-language comparison if multiple datasets
    if len(all_stats) > 1:
        compare_languages(all_stats)
    
    # Save to JSON
    # Convert Counter objects to dicts for JSON serialization
    output_stats = {}
    for lang, stats in all_stats.items():
        output_stats[lang] = {
            k: dict(v) if isinstance(v, Counter) else v 
            for k, v in stats.items()
            if k not in ['answer_lengths', 'question_lengths']  # Skip raw data
        }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Statistics saved to {args.output}")

if __name__ == "__main__":
    main()
