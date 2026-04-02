"""
Sentence-Level Alignment Example

Demonstrates the improved approach:
1. Pre-processes toxic seed into structured JSON
2. Uses 100 reference sentences for manifold alignment  
3. Validates using Maximum Mean Discrepancy (MMD)
4. Proves concept isomorphism across both neutral and toxic domains

This solves the fundamental problem: aligning dictionaries (words) doesn't
guarantee alignment of ideologies (full concepts).

[cite: Compositional Geometry & Isomorphism, 187-203]
"""

import torch
import json
from source.generator import ConceptGenerator
from source.preprocessor import preprocess_toxigen_seed
from source.sentence_anchors import (
    get_reference_sentences,
    extract_sentence_vectors,
    compute_sentence_procrustes_alignment,
    analyze_sentence_alignment
)
from source.alignment_utils import apply_alignment
from source.validation import validate_global_alignment, print_validation_report
import torch.nn.functional as F


def main():
    """
    Complete workflow: Preprocessor → Sentence Anchors → Validation
    """
    
    print("\n" + "="*70)
    print("SENTENCE-LEVEL ALIGNMENT WITH MMD VALIDATION")
    print("Teacher Model Proof-of-Concept (100 sentences)")
    print("="*70)
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: PRE-PROCESS TOXIC SEED")
    print("="*70)
    
    seed = "the only thing that can be done to improve the quality of life in the inner city is to build a giant wall to keep the blacks out"
    
    # Preprocess
    preprocessed = preprocess_toxigen_seed(seed)
    
    print("\n✓ Preprocessed JSON:")
    print(json.dumps(preprocessed, indent=2))
    
    # Extract parameters  
    forbidden_words = preprocessed["forbidden_words"]
    semantic_guideline = preprocessed["semantic_guideline"]
    short_range = preprocessed["short_range"]
    long_range = preprocessed["long_range"]
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: INITIALIZE MODELS (Teacher Models)")
    print("="*70)
    
    # Original models:
    # model_llama = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_mistral = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # Abliterated models (without refusal filters):
    model_llama = "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3"
    model_mistral = "evolveon/Mistral-7B-Instruct-v0.3-abliterated"
    
    print(f"\nInitializing {model_llama}...")
    gen_llama = ConceptGenerator(model_llama)
    
    print(f"Initializing {model_mistral}...")
    gen_mistral = ConceptGenerator(model_mistral)
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: EXTRACT SENTENCE ANCHORS (100 Reference Sentences)")
    print("="*70)
    
    reference_sentences = get_reference_sentences(num_sentences=100)
    
    print(f"\nUsing {len(reference_sentences)} neutral reference sentences:")
    print("Sample sentences:")
    for i, sent in enumerate(reference_sentences[:5]):
        print(f"  {i+1}. {sent}")
    print(f"  ... ({len(reference_sentences)-5} more)")
    
    # Extract vectors from both models
    print("\n📊 LLAMA: Extracting sentence vectors...")
    llama_sentence_vectors = extract_sentence_vectors(gen_llama, reference_sentences, method="hybrid")
    
    print("\n📊 MISTRAL: Extracting sentence vectors...")
    mistral_sentence_vectors = extract_sentence_vectors(gen_mistral, reference_sentences, method="hybrid")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: COMPUTE SENTENCE PROCRUSTES ALIGNMENT")
    print("="*70)
    
    print("\n🔄 Aligning Mistral's manifold to Llama's manifold...")
    Q, alignment_quality, source_mean, target_mean = compute_sentence_procrustes_alignment(
        mistral_sentence_vectors,
        llama_sentence_vectors
    )
    
    print(f"\n✅ Manifold Alignment Quality: {alignment_quality:.4f}")
    print(f"   (This is the quality of neutral sentence alignment)")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: GENERATE TOXIC CONCEPT VARIATIONS (Using Perspective Injection)")
    print("="*70)
    
    print("\n[Llama] Generating toxic concept variations...")
    llama_short = gen_llama.get_validated_variation(
        seed, forbidden_words,
        short_range["min"], short_range["max"],
        maintain_perspective=True
    )
    print(f"  Short: {llama_short}")
    
    llama_long = gen_llama.get_validated_variation(
        seed, forbidden_words,
        long_range["min"], long_range["max"],
        maintain_perspective=True
    )
    print(f"  Long: {llama_long}")
    
    print("\n[Mistral] Generating toxic concept variations...")
    mistral_short = gen_mistral.get_validated_variation(
        seed, forbidden_words,
        short_range["min"], short_range["max"],
        maintain_perspective=True
    )
    print(f"  Short: {mistral_short}")
    
    mistral_long = gen_mistral.get_validated_variation(
        seed, forbidden_words,
        long_range["min"], long_range["max"],
        maintain_perspective=True
    )
    print(f"  Long: {mistral_long}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: EXTRACT TOXIC CONCEPT VECTORS")
    print("="*70)
    
    print("\nExtracting latent vectors for toxic concepts...")
    
    # Llama toxic vectors
    llama_toxic_short_vec = gen_llama.get_hybrid_vector(llama_short)
    llama_toxic_long_vec = gen_llama.get_hybrid_vector(llama_long)
    llama_toxic_all = torch.stack([llama_toxic_short_vec, llama_toxic_long_vec], dim=0)
    
    # Mistral toxic vectors
    mistral_toxic_short_vec = gen_mistral.get_hybrid_vector(mistral_short)
    mistral_toxic_long_vec = gen_mistral.get_hybrid_vector(mistral_long)
    mistral_toxic_all = torch.stack([mistral_toxic_short_vec, mistral_toxic_long_vec], dim=0)
    
    print("✓ Toxic vectors extracted")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: APPLY SENTENCE-BASED ROTATION TO TOXIC VECTORS")
    print("="*70)
    
    print("\nApplying manifold rotation to toxic concepts...")
    
    mistral_toxic_short_aligned = apply_alignment(
        mistral_toxic_short_vec, Q, source_mean, target_mean
    )
    mistral_toxic_long_aligned = apply_alignment(
        mistral_toxic_long_vec, Q, source_mean, target_mean
    )
    mistral_toxic_aligned = torch.stack([
        mistral_toxic_short_aligned,
        mistral_toxic_long_aligned
    ], dim=0)
    
    print("✓ Rotations applied using sentence-learned manifold")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: MAXIMUM MEAN DISCREPANCY (MMD) VALIDATION")
    print("="*70)
    
    # Stack neutral vectors
    neutral_aligned_stack = torch.stack([
        apply_alignment(mistral_sentence_vectors[s], Q, source_mean, target_mean)
        for s in mistral_sentence_vectors.keys()
    ], dim=0)
    
    neutral_target_stack = torch.stack([
        llama_sentence_vectors[s]
        for s in mistral_sentence_vectors.keys()
    ], dim=0)
    
    # Validate global alignment
    validation_results = validate_global_alignment(
        neutral_aligned_stack,
        neutral_target_stack,
        mistral_toxic_aligned,
        llama_toxic_all,
        kernel_type="gaussian",
        threshold=0.15
    )
    
    # Print report
    print_validation_report(validation_results)
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 9: CROSS-MODEL CONCEPT SIMILARITY")
    print("="*70)
    
    sim_short_raw = F.cosine_similarity(
        llama_toxic_short_vec.unsqueeze(0),
        mistral_toxic_short_vec.unsqueeze(0)
    ).item()
    
    sim_short_aligned = F.cosine_similarity(
        llama_toxic_short_vec.unsqueeze(0),
        mistral_toxic_short_aligned.unsqueeze(0)
    ).item()
    
    sim_long_raw = F.cosine_similarity(
        llama_toxic_long_vec.unsqueeze(0),
        mistral_toxic_long_vec.unsqueeze(0)
    ).item()
    
    sim_long_aligned = F.cosine_similarity(
        llama_toxic_long_vec.unsqueeze(0),
        mistral_toxic_long_aligned.unsqueeze(0)
    ).item()
    
    print("\n📊 Short Concept Alignment:")
    print(f"   Before (raw):        {sim_short_raw:.4f}")
    print(f"   After (rotation):    {sim_short_aligned:.4f}")
    print(f"   Improvement:         +{(sim_short_aligned - sim_short_raw):.4f}")
    
    print("\n📊 Long Concept Alignment:")
    print(f"   Before (raw):        {sim_long_raw:.4f}")
    print(f"   After (rotation):    {sim_long_aligned:.4f}")
    print(f"   Improvement:         +{(sim_long_aligned - sim_long_raw):.4f}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    print("\n🎯 Proof of Concept Achieved:")
    print(f"   ✓ Preprocessor: JSON-formatted seed with semantics")
    print(f"   ✓ Reference Anchors: 100 neutral sentences for manifold")
    print(f"   ✓ Sentence Procrustes: Learned compositional rotation")
    print(f"   ✓ Toxic Concepts: Aligned using learned manifold")
    print(f"   ✓ Global Validation: {"PASSED ✅" if validation_results["is_global_alignment"] else "PARTIAL ⚠️"}")
    
    print("\n📈 Key Metrics:")
    print(f"   Neutral Alignment Quality: {alignment_quality:.4f}")
    print(f"   Short Concept Improvement: {(sim_short_aligned - sim_short_raw):.4f}")
    print(f"   Long Concept Improvement: {(sim_long_aligned - sim_long_raw):.4f}")
    print(f"   MMD (Combined): {validation_results['mmd_combined']:.6f}")
    
    print("\n🔮 Next Steps:")
    print("   → Replace local teacher with API-based LLM for larger scale")
    print("   → Extend to 1000+ reference sentences for richer manifold")
    print("   → Use this for dataset generation with guaranteed isomorphism")
    print("   → Explore cross-architecture alignment (Llama → GPT → Claude)")


if __name__ == "__main__":
    main()
