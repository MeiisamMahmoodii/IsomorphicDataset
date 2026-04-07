"""
Generate pre-computed alignment dataset from pre-extracted forbidden words.

This script reads a CSV with (original_text, forbidden_words) and generates
rewrites at different word lengths. Results are stored in a CSV for later
alignment testing, avoiding repeated generation during actual alignment tests.

Workflow:
  1. extract_forbidden_words_dataset.py → forbidden_words_dataset.csv
  2. generate_alignment_dataset.py --input forbidden_words_dataset.csv

Usage:
    python generate_alignment_dataset.py --input forbidden_words_dataset.csv --lengths 5-10 15-20
"""

import csv
import argparse
from typing import List, Tuple, Dict
from datetime import datetime
import warnings
import sys
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

from source.generator import ConceptGenerator

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    "llama": "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
    "mistral": "evolveon/Mistral-7B-Instruct-v0.3-abliterated"
}

DEFAULT_LENGTHS = [(5, 10), (15, 20), (20, 30)]
OUTPUT_CSV = "alignment_dataset.csv"


# =============================================================================
# LOAD FORBIDDEN WORDS DATASET
# =============================================================================

def load_forbidden_words_dataset(input_file: str) -> List[Dict[str, str]]:
    """
    Load dataset with original text and forbidden words.
    
    Args:
        input_file (str): Path to CSV with columns: original_text, forbidden_words
        
    Returns:
        List[Dict]: List of rows with original_text and forbidden_words
    """
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Loading forbidden words dataset: {input_file}\n")
    
    rows = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        print(f"✅ Loaded {len(rows)} entries from {input_file}\n")
        
        # Show sample
        if rows:
            print("Sample entry:")
            sample = rows[0]
            print(f"  Original: {sample.get('original_text', '')[:60]}...")
            print(f"  Forbidden: {sample.get('forbidden_words', 'N/A')}\n")
        
        return rows
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)


# =============================================================================
# GENERATE DATASET
# =============================================================================

def generate_dataset(
    input_file: str,
    length_ranges: List[Tuple[int, int]] = None,
    output_file: str = OUTPUT_CSV
):
    """
    Generate alignment dataset from forbidden words dataset.
    
    Args:
        input_file (str): CSV with original_text and forbidden_words columns
        length_ranges (List[Tuple[int, int]]): List of (min, max) word counts
        output_file (str): Output CSV file path
    """
    if length_ranges is None:
        length_ranges = DEFAULT_LENGTHS
    
    print("=" * 80)
    print("GENERATE ALIGNMENT DATASET WITH FORBIDDEN WORDS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load forbidden words dataset
    dataset_rows = load_forbidden_words_dataset(input_file)
    print(f"📋 Processing {len(dataset_rows)} entries")
    print(f"📏 Length ranges: {length_ranges}")
    print(f"🤖 Models: {list(MODELS.keys())}")
    print(f"💾 Output: {output_file}\n")
    
    # Initialize models
    print("Initializing models...")
    print("-" * 80)
    generators = {}
    for model_name, model_id in MODELS.items():
        print(f"Loading {model_name}...")
        try:
            generators[model_name] = ConceptGenerator(model_id)
            print(f"  ✅ {model_name} loaded\n")
        except Exception as e:
            print(f"  ❌ Error loading {model_name}: {e}\n")
    
    if not generators:
        print("❌ No models loaded. Exiting.")
        return
    
    # Generate dataset
    print("\n" + "=" * 80)
    print("GENERATING REWRITES WITH FORBIDDEN WORDS")
    print("=" * 80 + "\n")
    
    rows = []
    total_rewrites = 0
    
    for sent_idx, entry in enumerate(dataset_rows, 1):
        toxic_sent = entry.get('original_text', '')
        forbidden_str = entry.get('forbidden_words', '')
        forbidden_list = [w.strip() for w in forbidden_str.split('|') if w.strip()] if forbidden_str else []
        
        print(f"[{sent_idx}/{len(dataset_rows)}] Original: {toxic_sent[:50]}...")
        print(f"   Forbidden: {forbidden_list}")
        
        for min_words, max_words in length_ranges:
            for model_name, generator in generators.items():
                try:
                    # Generate rewrite with forbidden words constraint
                    prompt = f"Rewrite this statement in a more constructive way: {toxic_sent}"
                    response = generator.get_validated_variation(
                        seed_sentence=prompt,
                        forbidden_words=forbidden_list,
                        min_words=min_words,
                        max_words=max_words,
                        max_retries=3,
                        maintain_perspective=False
                    )
                    
                    if response:
                        # Count words
                        word_count = len(response.split())
                        
                        # Store row
                        rows.append({
                            'original_text': toxic_sent,
                            'forbidden_words': forbidden_str,
                            'rewritten_text': response,
                            'target_length': f"{min_words}-{max_words}",
                            'actual_length': word_count,
                            'model': model_name,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        total_rewrites += 1
                        print(f"  ✅ {model_name} ({min_words}-{max_words}): {response[:50]}...")
                    else:
                        print(f"  ❌ {model_name} ({min_words}-{max_words}): Failed to generate")
                        
                except Exception as e:
                    print(f"  ❌ {model_name} ({min_words}-{max_words}): Error - {str(e)[:50]}")
        
        print()
    
    # Save to CSV
    print("\n" + "=" * 80)
    print("SAVING DATASET")
    print("=" * 80 + "\n")
    
    if rows:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ Dataset saved: {output_file}")
        print(f"   Total rewrites: {total_rewrites}")
        print(f"   Rows: {len(rows)}")
        print(f"   Unique toxic sentences: {len(set(r['original_text'] for r in rows))}")
    else:
        print("❌ No data generated")
    
    print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alignment dataset from forbidden words CSV"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file with original_text and forbidden_words columns'
    )
    parser.add_argument(
        '--lengths',
        nargs='+',
        default=['5-10', '15-20', '20-30'],
        help='Word length ranges, e.g., "5-10 15-20 20-30" (default: 5-10 15-20 20-30)'
    )
    parser.add_argument(
        '--output',
        default=OUTPUT_CSV,
        help=f'Output CSV file (default: {OUTPUT_CSV})'
    )
    
    args = parser.parse_args()
    
    # Parse length ranges
    try:
        length_ranges = []
        for length_spec in args.lengths:
            min_w, max_w = map(int, length_spec.split('-'))
            length_ranges.append((min_w, max_w))
    except ValueError:
        print(f"❌ Invalid length format. Use 'MIN-MAX' e.g., '5-10'")
        sys.exit(1)
    
    # Generate dataset
    generate_dataset(
        input_file=args.input,
        length_ranges=length_ranges,
        output_file=args.output
    )
