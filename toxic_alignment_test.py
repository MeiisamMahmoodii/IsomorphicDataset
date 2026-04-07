"""
Toxic Dataset Alignment Test

Load toxic sentences from ToxiGen dataset, have both models rewrite them,
then calculate alignment between the rewrites.

This tests the core hypothesis:
"Can we align models when they're both rewriting the same toxic content?"
"""

import torch
import json
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

from source.generator import ConceptGenerator, generate_validated_response
from source.alignment_utils import calculate_procrustes_rotation, apply_alignment_batch
from source.semantic_judge import SemanticJudge
import torch.nn.functional as F

try:
    from datasets import load_dataset
    TOXIGEN_AVAILABLE = True
except ImportError:
    TOXIGEN_AVAILABLE = False
    print("⚠️ Warning: datasets library not available. Install with: pip install datasets")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    "llama": "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
    "mistral": "evolveon/Mistral-7B-Instruct-v0.3-abliterated"
}

NUM_TOXIC_SAMPLES = 50  # Number of toxic sentences to test
REWRITE_ATTEMPTS = 3    # Number of rewrites per sentence per model


# =============================================================================
# LOAD TOXIC DATASET
# =============================================================================

def load_toxigen_sentences(num_samples: int = 50) -> List[str]:
    """
    Load toxic sentences from ToxiGen dataset.
    
    Args:
        num_samples (int): Number of sentences to load
        
    Returns:
        List[str]: Toxic sentences
    """
    if not TOXIGEN_AVAILABLE:
        print("❌ ToxiGen dataset not available. Using fallback toxic sentences...\n")
        return get_fallback_toxic_sentences(num_samples)
    
    print(f"📥 Loading ToxiGen dataset ({num_samples} samples)...\n")
    
    try:
        dataset = load_dataset("toxigen", "train", split="train")
        
        # Extract sentences from the dataset
        toxic_sentences = []
        for i, example in enumerate(dataset):
            if len(toxic_sentences) >= num_samples:
                break
            
            # ToxiGen has 'text' field with toxic content
            if 'text' in example:
                toxic_sentences.append(example['text'])
            elif 'prompt' in example:
                toxic_sentences.append(example['prompt'])
        
        print(f"✅ Loaded {len(toxic_sentences)} toxic sentences from ToxiGen\n")
        
        # Show samples
        print("Sample toxic sentences from dataset:")
        for i, sent in enumerate(toxic_sentences[:3], 1):
            print(f"  {i}. {sent[:70]}...\n")
        
        return toxic_sentences
    
    except Exception as e:
        print(f"⚠️ Error loading ToxiGen: {e}")
        print("Using fallback sentences...\n")
        return get_fallback_toxic_sentences(num_samples)


def get_fallback_toxic_sentences(num_samples: int = 50) -> List[str]:
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
    
    return fallback[:num_samples]


# =============================================================================
# STEP 1: GENERATE REWRITES
# =============================================================================

def generate_rewrites(
    toxic_sentences: List[str],
    gen_llama: ConceptGenerator,
    gen_mistral: ConceptGenerator,
    attempts_per_sentence: int = 3
) -> Tuple[List[str], List[str]]:
    """
    Generate rewrites of toxic sentences from both models.
    
    Args:
        toxic_sentences: List of toxic sentences
        gen_llama: Llama generator
        gen_mistral: Mistral generator
        attempts_per_sentence: How many rewrites per sentence
        
    Returns:
        Tuple of (llama_rewrites, mistral_rewrites)
    """
    print("\n" + "="*80)
    print("STEP 1: GENERATE REWRITES FROM TOXIC SENTENCES")
    print("="*80)
    
    print(f"\nGenerating {attempts_per_sentence} rewrites per sentence\n")
    
    llama_rewrites = []
    mistral_rewrites = []
    
    for i, toxic_sent in enumerate(toxic_sentences, 1):
        print(f"\n[{i}/{len(toxic_sentences)}] Toxic: {toxic_sent[:60]}...")
        
        # Generate Llama rewrites
        for attempt in range(attempts_per_sentence):
            prompt = f"Rewrite this statement in a more constructive way: {toxic_sent}"
            response = generate_validated_response(
                seed_sentence=prompt,
                forbidden_words=[],
                min_words=5,
                max_words=10,
                model=gen_llama.model,
                tokenizer=gen_llama.tokenizer,
                max_retries=3,
                maintain_perspective=False
            )
            if response:
                llama_rewrites.append(response)
                print(f"  Llama #{attempt+1}: {response[:50]}...")
        
        # Generate Mistral rewrites
        for attempt in range(attempts_per_sentence):
            prompt = f"Rewrite this statement in a more constructive way: {toxic_sent}"
            response = generate_validated_response(
                seed_sentence=prompt,
                forbidden_words=[],
                min_words=5,
                max_words=10,
                model=gen_mistral.model,
                tokenizer=gen_mistral.tokenizer,
                max_retries=3,
                maintain_perspective=False
            )
            if response:
                mistral_rewrites.append(response)
                print(f"  Mistral #{attempt+1}: {response[:50]}...")
    
    print(f"\n✅ Generated:")
    print(f"   Llama rewrites: {len(llama_rewrites)}")
    print(f"   Mistral rewrites: {len(mistral_rewrites)}")
    
    return llama_rewrites, mistral_rewrites


# =============================================================================
# STEP 2: EXTRACT VECTORS
# =============================================================================

def extract_rewrite_vectors(
    rewrites: List[str],
    gen_model: ConceptGenerator,
    model_name: str
) -> torch.Tensor:
    """
    Extract latent vectors from rewrites.
    
    Args:
        rewrites: List of rewritten texts
        gen_model: Generator to extract vectors from
        model_name: Name for logging
        
    Returns:
        Tensor of shape [num_rewrites, hidden_dim]
    """
    print(f"\n📊 Extracting vectors from {model_name} rewrites...")
    
    vectors = []
    for i, rewrite in enumerate(rewrites, 1):
        vec = gen_model.get_latent_vector(rewrite)
        vectors.append(vec)
        
        if i % max(1, len(rewrites)//5) == 0:
            print(f"  ✓ Processed {i}/{len(rewrites)}")
    
    manifold = torch.stack(vectors)
    print(f"✅ Extracted {len(rewrites)} vectors: {manifold.shape}")
    
    return manifold


# =============================================================================
# STEP 3: CALCULATE ALIGNMENT
# =============================================================================

def calculate_toxic_alignment(
    llama_vectors: torch.Tensor,
    mistral_vectors: torch.Tensor
) -> Dict:
    """
    Calculate alignment between Llama and Mistral rewrites.
    
    Args:
        llama_vectors: [num_rewrites, hidden_dim]
        mistral_vectors: [num_rewrites, hidden_dim]
        
    Returns:
        Dict with alignment metrics
    """
    print("\n" + "="*80)
    print("STEP 2: CALCULATE ALIGNMENT")
    print("="*80)
    
    # Ensure same number of vectors
    min_len = min(len(llama_vectors), len(mistral_vectors))
    llama_vectors = llama_vectors[:min_len]
    mistral_vectors = mistral_vectors[:min_len]
    
    print(f"\nUsing {min_len} aligned rewrite pairs")
    
    # Compute Procrustes rotation
    Q, source_mean, target_mean = calculate_procrustes_rotation(
        mistral_vectors, llama_vectors
    )
    
    # Apply alignment
    aligned_mistral = apply_alignment_batch(
        mistral_vectors, Q, source_mean, target_mean
    )
    
    # Calculate quality metrics
    cosine_sims = F.cosine_similarity(aligned_mistral, llama_vectors)
    alignment_quality = cosine_sims.mean().item()
    
    results = {
        "num_pairs": min_len,
        "Q": Q,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "alignment_quality": alignment_quality,
        "mean_cosine": alignment_quality,
        "std_cosine": cosine_sims.std().item(),
        "min_cosine": cosine_sims.min().item(),
        "max_cosine": cosine_sims.max().item(),
        "cosine_similarities": cosine_sims.cpu().numpy().tolist(),
    }
    
    print(f"\n✅ Alignment Quality: {alignment_quality:.4f}")
    print(f"   Mean Cosine:     {results['mean_cosine']:.4f}")
    print(f"   Std Dev:         {results['std_cosine']:.4f}")
    print(f"   Min:             {results['min_cosine']:.4f}")
    print(f"   Max:             {results['max_cosine']:.4f}")
    
    return results


# =============================================================================
# STEP 4: REFERENCE MODEL VALIDATION
# =============================================================================

def validate_with_reference_judge(
    llama_rewrites: List[str],
    mistral_rewrites: List[str],
    gen_llama: ConceptGenerator
) -> Dict:
    """
    Use Llama as judge to validate semantic equivalence of rewrites.
    
    Args:
        llama_rewrites: List of Llama rewrites
        mistral_rewrites: List of Mistral rewrites
        gen_llama: Llama generator for judge
        
    Returns:
        Dict with validation metrics
    """
    print("\n" + "="*80)
    print("STEP 3: VALIDATE WITH REFERENCE JUDGE (Llama)")
    print("="*80)
    
    judge = SemanticJudge(gen_llama.model, gen_llama.tokenizer)
    
    min_len = min(len(llama_rewrites), len(mistral_rewrites))
    print(f"\nEvaluating {min_len} rewrite pairs from judge's perspective\n")
    
    cosine_sims = []
    verdicts_passed = 0
    
    for i, (llama_text, mistral_text) in enumerate(zip(llama_rewrites[:min_len], mistral_rewrites[:min_len]), 1):
        metrics = judge.evaluate_isomorphism(llama_text, mistral_text, use_wasserstein=False)
        verdict = judge.apply_thresholds(metrics, cosine_threshold=0.85, distance_threshold=0.05)
        
        cosine_sims.append(metrics['cosine_similarity'])
        if verdict['passed']:
            verdicts_passed += 1
        
        if i % max(1, min_len//5) == 0:
            print(f"  ✓ Evaluated {i}/{min_len} pairs")
    
    pass_rate = verdicts_passed / min_len * 100
    
    results = {
        "num_pairs": min_len,
        "passed_verdicts": verdicts_passed,
        "pass_rate": pass_rate,
        "mean_cosine": np.mean(cosine_sims),
        "std_cosine": np.std(cosine_sims),
        "min_cosine": np.min(cosine_sims),
        "max_cosine": np.max(cosine_sims),
        "cosine_similarities": cosine_sims,
    }
    
    print(f"\n✅ Judge's Verdict:")
    print(f"   Passed: {verdicts_passed}/{min_len} ({pass_rate:.1f}%)")
    print(f"   Mean Cosine: {results['mean_cosine']:.4f}")
    print(f"   Std Dev: {results['std_cosine']:.4f}")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution: Load toxic data, generate rewrites, calculate alignment."""
    
    print("\n" + "="*80)
    print("TOXIC DATASET ALIGNMENT TEST")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ========================================================================
    print("Initializing models...")
    print("-" * 80)
    
    print(f"Loading {MODELS['llama'].split('/')[-1]}...")
    gen_llama = ConceptGenerator(MODELS["llama"])
    
    print(f"Loading {MODELS['mistral'].split('/')[-1]}...")
    gen_mistral = ConceptGenerator(MODELS["mistral"])
    
    print("✅ Both models loaded\n")
    
    # ========================================================================
    # Load toxic sentences
    toxic_sentences = load_toxigen_sentences(NUM_TOXIC_SAMPLES)
    
    # ========================================================================
    # Generate rewrites
    llama_rewrites, mistral_rewrites = generate_rewrites(
        toxic_sentences,
        gen_llama,
        gen_mistral,
        attempts_per_sentence=REWRITE_ATTEMPTS
    )
    
    # Skip if no rewrites generated
    if not llama_rewrites or not mistral_rewrites:
        print("\n❌ No rewrites generated. Exiting.")
        return
    
    # ========================================================================
    # Extract vectors
    print("\n" + "="*80)
    print("STEP 1b: EXTRACT VECTORS FROM REWRITES")
    print("="*80)
    
    llama_vectors = extract_rewrite_vectors(llama_rewrites, gen_llama, "Llama")
    mistral_vectors = extract_rewrite_vectors(mistral_rewrites, gen_mistral, "Mistral")
    
    # ========================================================================
    # Calculate alignment
    alignment_results = calculate_toxic_alignment(llama_vectors, mistral_vectors)
    
    # ========================================================================
    # Validate with judge
    judge_results = validate_with_reference_judge(
        llama_rewrites,
        mistral_rewrites,
        gen_llama
    )
    
    # ========================================================================
    # Final Report
    print("\n\n" + "="*80)
    print("FINAL REPORT: TOXIC DATASET ALIGNMENT")
    print("="*80)
    
    print(f"\n📊 Data Summary:")
    print(f"   Original toxic sentences: {len(toxic_sentences)}")
    print(f"   Llama rewrites: {len(llama_rewrites)}")
    print(f"   Mistral rewrites: {len(mistral_rewrites)}")
    print(f"   Aligned pairs: {alignment_results['num_pairs']}")
    
    print(f"\n📈 Procrustes Alignment Results:")
    print(f"   Quality (Mean Cosine): {alignment_results['alignment_quality']:.4f}")
    print(f"   Std Dev: {alignment_results['std_cosine']:.4f}")
    print(f"   Range: [{alignment_results['min_cosine']:.4f}, {alignment_results['max_cosine']:.4f}]")
    
    print(f"\n🏛️ Reference Judge Results (Llama):")
    print(f"   Pass Rate: {judge_results['pass_rate']:.1f}%")
    print(f"   Mean Cosine: {judge_results['mean_cosine']:.4f}")
    print(f"   Std Dev: {judge_results['std_cosine']:.4f}")
    
    print(f"\n🎯 Key Findings:")
    
    if alignment_results['alignment_quality'] > 0.85:
        print(f"   ✅ Strong mathematical alignment ({alignment_results['alignment_quality']:.4f})")
    elif alignment_results['alignment_quality'] > 0.70:
        print(f"   ⚠️ Moderate mathematical alignment ({alignment_results['alignment_quality']:.4f})")
    else:
        print(f"   ❌ Weak mathematical alignment ({alignment_results['alignment_quality']:.4f})")
    
    if judge_results['pass_rate'] > 80:
        print(f"   ✅ High semantic equivalence ({judge_results['pass_rate']:.1f}% pass rate)")
    elif judge_results['pass_rate'] > 60:
        print(f"   ⚠️ Moderate semantic equivalence ({judge_results['pass_rate']:.1f}% pass rate)")
    else:
        print(f"   ❌ Low semantic equivalence ({judge_results['pass_rate']:.1f}% pass rate)")
    
    combined_score = alignment_results['alignment_quality'] * (judge_results['pass_rate']/100)
    print(f"\n   Combined Score: {combined_score:.4f}")
    print(f"   (Procrustes × Judge Pass Rate)")
    
    print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
