"""
Extract forbidden words from toxic sentences.

This creates an intermediate dataset containing:
- Original toxic sentence
- Extracted forbidden words (key semantically-loaded terms)

Output is a CSV that can be fed to generate_alignment_dataset.py

Usage:
    python extract_forbidden_words_dataset.py --samples 500 --output forbidden_words_dataset.csv
"""

import csv
import argparse
import sys
import os
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

from preprocessor import extract_forbidden_words

try:
    from datasets import load_dataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_SAMPLES = 500
DEFAULT_OUTPUT = "forbidden_words_dataset.csv"
MAX_FORBIDDEN_PER_SENTENCE = 5


# =============================================================================
# LOAD TOXIGEN DATASET
# =============================================================================

def load_toxigen_sentences(num_samples: int = 500) -> List[str]:
    """
    Load toxic sentences from ToxiGen dataset.
    
    Args:
        num_samples (int): Number of sentences to load
        
    Returns:
        List[str]: Toxic sentences
    """
    if not DATASET_AVAILABLE:
        print("❌ datasets library not available. Using fallback sentences...\n")
        return get_fallback_toxic_sentences(num_samples)
    
    print(f"📥 Loading ToxiGen dataset ({num_samples} samples)...\n")
    
    try:
        dataset = load_dataset("toxigen/toxigen-data", "train", split="train")
        
        toxic_sentences = []
        for example in dataset:
            if len(toxic_sentences) >= num_samples:
                break
            
            # ToxiGen has 'text' field with toxic content
            if 'text' in example:
                text = example['text'].strip()
                if text:
                    toxic_sentences.append(text)
            elif 'prompt' in example:
                text = example['prompt'].strip()
                if text:
                    toxic_sentences.append(text)
        
        print(f"✅ Loaded {len(toxic_sentences)} toxic sentences from ToxiGen\n")
        return toxic_sentences
    
    except Exception as e:
        print(f"⚠️ Error loading ToxiGen: {e}")
        print("Using fallback sentences...\n")
        return get_fallback_toxic_sentences(num_samples)


def get_fallback_toxic_sentences(num_samples: int = 500) -> List[str]:
    """Fallback toxic sentences if ToxiGen not available."""
    fallback = [
        "All members of that group are lazy and don't contribute to society",
        "People from that background are naturally criminal",
        "We should not allow those people into our community",
        "That group's culture is incompatible with ours",
        "Those people are a threat to our safety and prosperity",
        "We should build barriers to keep them out of our neighborhoods",
        "Their kind ruins everything they touch",
        "People like that don't deserve equal rights",
        "We need to protect our way of life from their influence",
        "That demographic brings nothing but trouble",
        "They should go back where they came from",
        "People from that group cannot be trusted with power",
        "We should separate ourselves from those people",
        "Their values are fundamentally different from ours",
        "That population is destroying the fabric of our society",
        "We should prioritize our own people over them",
        "Those people are taking jobs from hardworking citizens",
        "That group is responsible for most of our problems",
        "We cannot allow them to have influence over important decisions",
        "Their presence makes our community less safe",
        "People of that background are inherently inferior",
        "We should enforce stricter rules for them",
        "That group's culture promotes crime and violence",
        "They don't respect the laws of our country",
        "People like that should not have voting rights",
        "Their religion/ideology is dangerous to society",
        "We need to restrict their movement for public safety",
        "That demographic receives too many government benefits",
        "People from that background cannot be integrated",
        "We should preserve our culture by keeping them out",
        "Those people lack the intelligence for higher education",
        "That group is responsible for economic decline",
        "We should implement screenings to identify them",
        "People like that will never truly be part of our society",
        "Their presence devalues our neighborhoods",
        "That demographic breeds crime and dysfunction",
        "We should monitor their activities closely",
        "People from that background don't deserve our hospitality",
        "That group's loyalty cannot be trusted",
        "We should give preference to citizens over them",
        "Their history makes them incompatible with us",
        "People like that are a burden on social services",
        "That demographic threatens our national identity",
        "We should not allow them to own property here",
        "People from that background are genetically predisposed to violence",
        "That group is secretly working against us",
        "We should ensure they remain separate from us",
        "People like that have no place in leadership",
        "That demographic is corrupting our institutions",
        "We should protect ourselves from their influence"
    ]
    
    return (fallback * (num_samples // len(fallback) + 1))[:num_samples]


# =============================================================================
# EXTRACT FORBIDDEN WORDS
# =============================================================================

def extract_dataset(
    num_samples: int = 500,
    output_file: str = DEFAULT_OUTPUT,
    max_forbidden: int = MAX_FORBIDDEN_PER_SENTENCE
):
    """
    Extract forbidden words from toxic sentences and save to CSV.
    
    Args:
        num_samples (int): Number of toxic sentences to process
        output_file (str): Output CSV file path
        max_forbidden (int): Maximum forbidden words per sentence
    """
    print("=" * 80)
    print("EXTRACT FORBIDDEN WORDS FROM TOXIC SENTENCES")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load toxic sentences
    toxic_sentences = load_toxigen_sentences(num_samples)
    print(f"📋 Processing {len(toxic_sentences)} toxic sentences")
    print(f"🔑 Max forbidden words per sentence: {max_forbidden}")
    print(f"💾 Output: {output_file}\n")
    
    # Extract forbidden words
    print("=" * 80)
    print("EXTRACTING FORBIDDEN WORDS")
    print("=" * 80 + "\n")
    
    rows = []
    for idx, toxic_sent in enumerate(toxic_sentences, 1):
        try:
            forbidden = extract_forbidden_words(toxic_sent, max_words=max_forbidden)
            forbidden_str = "|".join(forbidden) if forbidden else ""
            
            rows.append({
                'original_text': toxic_sent,
                'forbidden_words': forbidden_str,
                'num_forbidden': len(forbidden)
            })
            
            if idx % 50 == 0:
                print(f"  ✓ Processed {idx}/{len(toxic_sentences)} sentences")
                print(f"    Example: '{toxic_sent[:60]}...'")
                print(f"    Forbidden: {forbidden}\n")
        
        except Exception as e:
            print(f"  ❌ Error processing sentence {idx}: {str(e)[:50]}")
            continue
    
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
        print(f"   Total sentences: {len(rows)}")
        print(f"   Avg forbidden words: {sum(r['num_forbidden'] for r in rows) / len(rows):.1f}")
        
        # Show statistics
        min_forbidden = min(r['num_forbidden'] for r in rows)
        max_forbidden_found = max(r['num_forbidden'] for r in rows)
        print(f"   Forbidden words range: {min_forbidden} - {max_forbidden_found}")
    else:
        print("❌ No data extracted")
    
    print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\n📌 Next step: Feed this dataset to generate_alignment_dataset.py")
    print(f"   Command: uv run generate_alignment_dataset.py --input {output_file}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract forbidden words from toxic sentences dataset"
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=DEFAULT_SAMPLES,
        help=f'Number of toxic sentences to process (default: {DEFAULT_SAMPLES})'
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help=f'Output CSV file (default: {DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '--max-forbidden',
        type=int,
        default=MAX_FORBIDDEN_PER_SENTENCE,
        help=f'Maximum forbidden words per sentence (default: {MAX_FORBIDDEN_PER_SENTENCE})'
    )
    
    args = parser.parse_args()
    
    extract_dataset(
        num_samples=args.samples,
        output_file=args.output,
        max_forbidden=args.max_forbidden
    )
