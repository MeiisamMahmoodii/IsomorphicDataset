"""
Cross-Model Alignment Example
==============================

This example demonstrates the complete workflow for aligning concept vectors
between two different LLMs (Llama and Mistral) using Procrustes analysis.

The key insight: Different models have different "coordinate systems" for their
latent spaces. We use neutral anchor words to find the rotation matrix that
maps one model's space to another's, enabling true cross-model comparison.

[cite: Procrustes Analysis for LLM Alignment, 194]
"""

import torch
import torch.nn.functional as F
from source.generator import ConceptGenerator
from source.alignment_utils import calculate_procrustes_rotation, apply_alignment
from source.alignment import get_all_anchor_words


def main():
    """Complete step-by-step example of cross-model alignment."""
    
    print("="*70)
    print("CROSS-MODEL ALIGNMENT: BRIDGING LATENT SPACE GAPS")
    print("="*70)
    
    # Define the two models to align
    model_llama = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_mistral = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # Define a concept for case study
    seed_concept = "the only thing that can be done to improve the quality of life in the inner city is to build a giant wall to keep the blacks out"
    forbidden_words = ["blacks", "wall", "city"]
    
    print("\n" + "="*70)
    print("STEP 1: INITIALIZE MODELS")
    print("="*70)
    
    print(f"\nInitializing {model_llama}...")
    gen_llama = ConceptGenerator(model_llama)
    
    print(f"Initializing {model_mistral}...")
    gen_mistral = ConceptGenerator(model_mistral)
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: GENERATE ISOMORPHIC CONCEPT VARIATIONS")
    print("="*70)
    
    # First, generate WITHOUT perspective injection (baseline)
    print("\n[BASELINE] Generating without perspective enforcement...")
    print("\n[Llama] Generating short variation (5-10 words)...")
    llama_short_baseline = gen_llama.get_validated_variation(
        seed_concept, forbidden_words, min_words=5, max_words=10,
        maintain_perspective=False
    )
    print(f"Result: {llama_short_baseline}")
    
    print("\n[Mistral] Generating short variation (5-10 words)...")
    mistral_short_baseline = gen_mistral.get_validated_variation(
        seed_concept, forbidden_words, min_words=5, max_words=10,
        maintain_perspective=False
    )
    print(f"Result: {mistral_short_baseline}")
    
    # Then, generate WITH perspective injection (improved)
    print("\n[WITH PERSPECTIVE INJECTION] Enforcing semantic consistency...")
    print("\n[Llama] Generating with perspective enforcement...")
    llama_short = gen_llama.get_validated_variation(
        seed_concept, forbidden_words, min_words=5, max_words=10,
        maintain_perspective=True
    )
    print(f"Result: {llama_short}")
    
    print("\n[Mistral] Generating with perspective enforcement...")
    mistral_short = gen_mistral.get_validated_variation(
        seed_concept, forbidden_words, min_words=5, max_words=10,
        maintain_perspective=True
    )
    print(f"Result: {mistral_short}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: EXTRACT LATENT VECTORS (BEFORE ALIGNMENT)")
    print("="*70)
    
    print("\nExtracting concept vectors using hybrid method (mean pooling + last token)...")
    llama_concept_vec = gen_llama.get_hybrid_vector(llama_short)
    mistral_concept_vec = gen_mistral.get_hybrid_vector(mistral_short)
    
    print(f"Llama vector shape: {llama_concept_vec.shape}")
    print(f"Mistral vector shape: {mistral_concept_vec.shape}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: COMPUTE RAW CROSS-MODEL SIMILARITY (BEFORE ALIGNMENT)")
    print("="*70)
    
    raw_similarity = F.cosine_similarity(
        llama_concept_vec.unsqueeze(0), 
        mistral_concept_vec.unsqueeze(0)
    )
    
    print(f"\n❌ Raw Cross-Model Similarity: {raw_similarity.item():.4f}")
    print("   (Near zero because models developed different coordinate systems)")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: EXTRACT ANCHOR WORD VECTORS (Neutral Reference Points)")
    print("="*70)
    
    # Select a subset of anchor words for demonstration
    demo_anchors = [
        "mathematics", "oxygen", "geography", "velocity", "keyboard", "mountain",
        "science", "light", "water", "motion", "number", "color"
    ]
    
    print(f"\nUsing {len(demo_anchors)} neutral anchor words:")
    print(f"  {', '.join(demo_anchors)}")
    
    print("\nExtracting anchor vectors from Llama...")
    llama_anchors = {}
    for word in demo_anchors:
        try:
            llama_anchors[word] = gen_llama.get_hybrid_vector(word)
        except Exception as e:
            print(f"  ⚠️ Failed for '{word}': {e}")
    
    print(f"✅ Successfully extracted {len(llama_anchors)} anchor vectors")
    
    print("\nExtracting anchor vectors from Mistral...")
    mistral_anchors = {}
    for word in demo_anchors:
        try:
            mistral_anchors[word] = gen_mistral.get_hybrid_vector(word)
        except Exception as e:
            print(f"  ⚠️ Failed for '{word}': {e}")
    
    print(f"✅ Successfully extracted {len(mistral_anchors)} anchor vectors")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: SOLVE PROCRUSTES PROBLEM")
    print("="*70)
    
    # Get common anchors
    common_anchors = set(llama_anchors.keys()) & set(mistral_anchors.keys())
    print(f"\nUsing {len(common_anchors)} common anchor words")
    
    # Stack vectors
    mistral_stack = torch.stack([mistral_anchors[w] for w in common_anchors], dim=0)
    llama_stack = torch.stack([llama_anchors[w] for w in common_anchors], dim=0)
    
    print(f"Anchor matrix shapes: {mistral_stack.shape} -> {llama_stack.shape}")
    
    # Compute optimal rotation using SVD-based Procrustes
    print("\nSolving: Q = argmin ||Llama_anchors - Mistral_anchors @ Q||_F")
    Q, source_mean, target_mean = calculate_procrustes_rotation(mistral_stack, llama_stack)
    
    print(f"✅ Rotation matrix Q computed: {Q.shape}")
    print(f"   Source mean: {source_mean.shape}")
    print(f"   Target mean: {target_mean.shape}")
    
    # Verify alignment on anchor words
    print("\nVerifying alignment quality on anchor words...")
    anchor_similarities = []
    for word in list(common_anchors)[:5]:  # Show first 5
        aligned = apply_alignment(mistral_anchors[word], Q, source_mean, target_mean)
        sim = F.cosine_similarity(aligned.unsqueeze(0), llama_anchors[word].unsqueeze(0))
        anchor_similarities.append(sim.item())
        print(f"  '{word}': {sim.item():.4f}")
    
    mean_anchor_sim = sum(anchor_similarities) / len(anchor_similarities)
    print(f"\nMean anchor alignment quality: {mean_anchor_sim:.4f}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: APPLY ROTATION TO CONCEPT VECTOR")
    print("="*70)
    
    print("\nApplying Procrustes transformation to Mistral's concept vector...")
    print("  Formula: (vector - source_mean) @ Q + target_mean")
    
    aligned_mistral_concept_vec = apply_alignment(
        mistral_concept_vec, 
        Q, 
        source_mean, 
        target_mean
    )
    
    print(f"✅ Mistral concept vector aligned to Llama space")
    print(f"   Original shape: {mistral_concept_vec.shape}")
    print(f"   Aligned shape: {aligned_mistral_concept_vec.shape}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: VERIFY ALIGNMENT (AFTER)")
    print("="*70)
    
    aligned_similarity = F.cosine_similarity(
        llama_concept_vec.unsqueeze(0), 
        aligned_mistral_concept_vec.unsqueeze(0)
    )
    
    print(f"\n✅ Aligned Cross-Model Similarity: {aligned_similarity.item():.4f}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    improvement = aligned_similarity.item() - raw_similarity.item()
    
    print(f"\nBefore Alignment: {raw_similarity.item():.4f}")
    print(f"After Alignment:  {aligned_similarity.item():.4f}")
    print(f"Improvement:      +{improvement:.4f} ({improvement*100:.1f}%)")
    
    if improvement > 0:
        print("\n✨ SUCCESS: Models' latent spaces are now aligned!")
        print("   The Procrustes rotation successfully mapped Mistral's coordinate")
        print("   system to Llama's, enabling meaningful cross-model comparison.")
    else:
        print("\n⚠️ Note: Improvement may vary depending on:")
        print("   - Model architecture differences")
        print("   - Anchor word selection")
        print("   - Concept complexity")
    
    # ========================================================================
    print("\n" + "="*70)
    print("DETAILED INFORMATION")
    print("="*70)
    
    print(f"\nLlama Concept:")
    print(f"  Text: {llama_short}")
    print(f"  Vector shape: {llama_concept_vec.shape}")
    print(f"  Vector stats: mean={llama_concept_vec.mean():.4f}, std={llama_concept_vec.std():.4f}")
    
    print(f"\nMistral Concept (Original):")
    print(f"  Text: {mistral_short}")
    print(f"  Vector shape: {mistral_concept_vec.shape}")
    print(f"  Vector stats: mean={mistral_concept_vec.mean():.4f}, std={mistral_concept_vec.std():.4f}")
    
    print(f"\nMistral Concept (Aligned):")
    print(f"  Vector shape: {aligned_mistral_concept_vec.shape}")
    print(f"  Vector stats: mean={aligned_mistral_concept_vec.mean():.4f}, std={aligned_mistral_concept_vec.std():.4f}")
    
    print(f"\nRotation Matrix Q:")
    print(f"\nRotation Matrix Q:")
    print(f"  Shape: {Q.shape}")
    print(f"  Is orthogonal: {torch.allclose(Q @ Q.T, torch.eye(Q.shape[0]), atol=1e-4)}")
    print(f"  Determinant: {torch.det(Q).item():.4f} (should be ±1 for rotation)")
    
    # ========================================================================
    print(f"\n{'='*70}")
    print("PERSPECTIVE INJECTION IMPACT")
    print(f"{'='*70}")
    
    print("\nThis example demonstrates TWO strategies for semantic alignment:")
    print("\n1. WITHOUT PERSPECTIVE INJECTION (Baseline):")
    print(f"   Llama:    {llama_short_baseline}")
    print(f"   Mistral:  {mistral_short_baseline}")
    print("   Issue: Models may \"disagree\" and offer alternatives")
    
    print("\n2. WITH PERSPECTIVE INJECTION (Improved):")
    print(f"   Llama:    {llama_short}")
    print(f"   Mistral:  {mistral_short}")
    print("   Benefit: Models maintain consistent perspective for true isomorphism")
    
    print("\n📊 KEY INSIGHT:")
    print("   Perspective injection forces both models into the SAME semantic")
    print("   coordinate system by explicitly requiring them to maintain the")
    print("   original perspective. This BEFORE Procrustes alignment ensures:")
    print("   • Higher raw cross-model similarity (less need to rotate)")
    print("   • More semantically meaningful alignments")
    print("   • Better concept preservation across variations")


if __name__ == "__main__":
    main()
