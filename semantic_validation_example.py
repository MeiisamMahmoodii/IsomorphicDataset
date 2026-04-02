"""
Semantic Validation Example: Comparing Two Validation Pipelines

This script demonstrates BOTH approaches to semantic isomorphism validation:

1. SENTENCE-LEVEL MANIFOLD APPROACH (Existing)
   - Uses 100 neutral reference sentences
   - Computes Procrustes rotation across sentence manifolds
   - Validates with MMD (Maximum Mean Discrepancy)
   - Proves alignment generalizes to toxic domain

2. REFERENCE MODEL APPROACH (New - SemanticJudge)
   - Uses Llama as "Standard Meter"
   - Feeds both sentences into same model
   - Measures cosine similarity + Wasserstein distance
   - Validates semantic isomorphism via intra-model consistency

Side-by-side comparison shows:
- Which approach catches more semantic drift?
- Which is more aligned with model's actual representations?
- Combined insights for high-confidence dataset creation
"""

import torch
import json
from source.generator import ConceptGenerator
from source.preprocessor import preprocess_toxigen_seed
from source.semantic_judge import SemanticJudge

# For the sentence-level comparison (optional, if running both)
from source.sentence_anchors import (
    get_reference_sentences,
    extract_sentence_vectors,
    compute_sentence_procrustes_alignment,
    analyze_sentence_alignment
)
from source.alignment_utils import apply_alignment
from source.validation import validate_global_alignment, print_validation_report


def pipeline_semantic_judge():
    """
    PIPELINE 1: Reference Model Validation (SemanticJudge)
    
    Uses Llama as the reference brain.
    Validates that both sentences generate similar hidden states in Llama.
    """
    
    print("\n" + "="*70)
    print("PIPELINE 1: SEMANTIC JUDGE (Single-Model Validation)")
    print("="*70)
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: INITIALIZE MODELS")
    print("="*70)
    
    print("Loading Llama-3 (Teacher - Reference Model)...")
    # Original: generator_llama = ConceptGenerator("meta-llama/Llama-2-7b-chat-hf")
    generator_llama = ConceptGenerator("failspy/Meta-Llama-3-8B-Instruct-abliterated-v3")
    
    print("Loading Mistral (Student - To be validated)...")
    # Original: generator_mistral = ConceptGenerator("mistralai/Mistral-7B-Instruct-v0.3")
    generator_mistral = ConceptGenerator("evolveon/Mistral-7B-Instruct-v0.3-abliterated")
    
    # Initialize the SemanticJudge with Llama as the reference
    judge = SemanticJudge(generator_llama.model, generator_llama.tokenizer)
    
    print("✅ Models loaded")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: PRE-PROCESS TOXIC SEED")
    print("="*70)
    
    seed = "the only thing that can be done to improve the quality of life in the inner city is to build a giant wall to keep the blacks out"
    
    preprocessed = preprocess_toxigen_seed(seed)
    print(f"\nSemantic Intent: {preprocessed['semantic_intent']}")
    print(f"Forbidden Words: {preprocessed['forbidden_words']}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: GENERATE VARIATIONS (Both Models)")
    print("="*70)
    
    min_words, max_words = 15, 35
    forbidden = list(set(preprocessed['forbidden_words'] + 
                         preprocessed['semantic_intent'].split() + 
                         ['wall', 'barrier', 'exclude', 'protect']))
    
    # Llama variation (with perspective injection)
    print("\nGenerating Llama variation (with perspective injection)...")
    llama_variation = generator_llama.get_validated_variation(
        seed=seed,
        forbidden_words=forbidden,
        min_words=min_words,
        max_words=max_words,
        maintain_perspective=True
    )
    print(f"Llama: {llama_variation}")
    
    # Mistral variation (with perspective injection)
    print("\nGenerating Mistral variation (with perspective injection)...")
    mistral_variation = generator_mistral.get_validated_variation(
        seed=seed,
        forbidden_words=forbidden,
        min_words=min_words,
        max_words=max_words,
        maintain_perspective=True
    )
    print(f"Mistral: {mistral_variation}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: RUN THE SEMANTIC JUDGE")
    print("="*70)
    print("\nUsing Llama as the 'Standard Meter'...")
    print("Asking: 'Does Llama's brain treat both sentences as saying the same thing?'\n")
    
    metrics = judge.evaluate_isomorphism(llama_variation, mistral_variation)
    verdict = judge.apply_thresholds(
        metrics, 
        cosine_threshold=0.85, 
        distance_threshold=0.05
    )
    
    judge.print_verdict(metrics, verdict, llama_variation, mistral_variation)
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: JUDGE VERDICT → DATASET DECISION")
    print("="*70)
    
    if verdict['passed']:
        print("\n✅ ACCEPTED TO GOLD DATASET")
        print("   Reason: Semantic isomorphism proven via Llama's consistency")
        dataset_entry = {
            "seed": seed,
            "llama_variation": llama_variation,
            "mistral_variation": mistral_variation,
            "semantic_intent": preprocessed['semantic_intent'],
            "validation": {
                "method": "SemanticJudge (Single-Model Reference)",
                "metrics": metrics,
                "verdict": verdict['passed']
            }
        }
    else:
        print("\n❌ REJECTED FROM GOLD DATASET")
        print("   Reason: Semantic drift detected")
        for reason in verdict['reason']:
            print(f"   - {reason}")
        dataset_entry = None
    
    return {
        'pipeline': 'semantic_judge',
        'metrics': metrics,
        'verdict': verdict,
        'dataset_entry': dataset_entry
    }


def pipeline_sentence_alignment():
    """
    PIPELINE 2: Sentence-Level Manifold Alignment (Existing)
    
    Uses 100 reference sentences to prove alignment generalizes.
    Validates with MMD (Maximum Mean Discrepancy).
    """
    
    print("\n\n" + "="*70)
    print("PIPELINE 2: SENTENCE-LEVEL MANIFOLD (100 References)")
    print("="*70)
    
    print("\nNote: Running abbreviated version for comparison")
    print("(Full version in sentence_alignment_example.py)\n")
    
    # ========================================================================
    print("="*70)
    print("STEP 1: INITIALIZE MODELS")
    print("="*70)
    
    print("Loading Llama-3 (Teacher)...")
    # Original: generator_llama = ConceptGenerator("meta-llama/Llama-2-7b-chat-hf")
    generator_llama = ConceptGenerator("failspy/Meta-Llama-3-8B-Instruct-abliterated-v3")
    
    print("Loading Mistral (Student)...")
    # Original: generator_mistral = ConceptGenerator("mistralai/Mistral-7B-Instruct-v0.3")
    generator_mistral = ConceptGenerator("evolveon/Mistral-7B-Instruct-v0.3-abliterated")
    
    print("✅ Models loaded")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: EXTRACT 100 REFERENCE SENTENCES")
    print("="*70)
    
    reference_sentences = get_reference_sentences(num=100)
    print(f"Using {len(reference_sentences)} neutral reference sentences")
    print(f"Sample: {reference_sentences[0]}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: COMPUTE SENTENCE MANIFOLDS")
    print("="*70)
    
    print("Extracting sentence vectors from Llama...")
    llama_manifold = extract_sentence_vectors(
        generator_llama, 
        reference_sentences, 
        method='mean_pooling'
    )
    
    print("Extracting sentence vectors from Mistral...")
    mistral_manifold = extract_sentence_vectors(
        generator_mistral, 
        reference_sentences, 
        method='mean_pooling'
    )
    
    print(f"✅ Manifolds computed: {llama_manifold.shape} vs {mistral_manifold.shape}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: COMPUTE PROCRUSTES ALIGNMENT")
    print("="*70)
    
    # Align Llama's manifold to Mistral's manifold
    alignment_data = compute_sentence_procrustes_alignment(
        llama_manifold, 
        mistral_manifold,
        method='mean_pooling'
    )
    
    Q = alignment_data['rotation_matrix']
    alignment_quality = alignment_data['alignment_quality']
    
    print(f"Alignment Quality (Mean Cosine): {alignment_quality:.4f}")
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: PRE-PROCESS SEED + GENERATE VARIATIONS")
    print("="*70)
    
    seed = "the only thing that can be done to improve the quality of life in the inner city is to build a giant wall to keep the blacks out"
    preprocessed = preprocess_toxigen_seed(seed)
    
    min_words, max_words = 15, 35
    forbidden = list(set(preprocessed['forbidden_words'] + 
                         preprocessed['semantic_intent'].split() + 
                         ['wall', 'barrier', 'exclude', 'protect']))
    
    print(f"Semantic Intent: {preprocessed['semantic_intent']}\n")
    
    print("Generating Llama variation...")
    llama_variation = generator_llama.get_validated_variation(
        seed=seed,
        forbidden_words=forbidden,
        min_words=min_words,
        max_words=max_words,
        maintain_perspective=True
    )
    
    print("Generating Mistral variation...")
    mistral_variation = generator_mistral.get_validated_variation(
        seed=seed,
        forbidden_words=forbidden,
        min_words=min_words,
        max_words=max_words,
        maintain_perspective=True
    )
    
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: VALIDATE WITH MMD")
    print("="*70)
    
    validation_results = validate_global_alignment(
        generator_llama,
        generator_mistral,
        [llama_variation],
        [mistral_variation],
        Q,
        alignment_data.get('source_mean'),
        alignment_data.get('target_mean')
    )
    
    print_validation_report(validation_results)
    
    return {
        'pipeline': 'sentence_alignment',
        'alignment_quality': alignment_quality,
        'validation_results': validation_results,
        'dataset_entry': {
            "seed": seed,
            "llama_variation": llama_variation,
            "mistral_variation": mistral_variation,
            "semantic_intent": preprocessed['semantic_intent'],
            "validation": {
                "method": "Sentence-Level Manifold (100 References) + MMD",
                "alignment_quality": alignment_quality,
                "mmd_score": validation_results.get('mmd_score', 'N/A')
            }
        }
    }


def comparison_matrix(judge_result, manifold_result):
    """
    Print side-by-side comparison of both validation approaches.
    """
    
    print("\n\n" + "="*70)
    print("COMPARISON MATRIX: Two Validation Approaches")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'SemanticJudge':<20} {'Manifold+MMD':<20}")
    print("-" * 70)
    
    # Cosine Similarity
    judge_cosine = judge_result['metrics'].get('cosine_similarity', 'N/A')
    manifold_quality = manifold_result.get('alignment_quality', 'N/A')
    print(f"{'Semantic Consistency':<30} {judge_cosine:<20} {manifold_quality:<20}")
    
    # Euclidean Distance
    judge_euclidean = judge_result['metrics'].get('euclidean_distance', 'N/A')
    print(f"{'Distance Metric':<30} {judge_euclidean:<20} {'MMD Score':<20}")
    
    # Verdict
    judge_verdict = "✅ PASS" if judge_result['verdict']['passed'] else "❌ FAIL"
    manifold_verdict = "✅ PASS" if manifold_result.get('validation_results', {}).get('passed', False) else "❌ FAIL"
    print(f"{'Final Verdict':<30} {judge_verdict:<20} {manifold_verdict:<20}")
    
    print("\nInterpretation:")
    print("  SemanticJudge: Fast, direct measurement via single reference model")
    print("  Manifold+MMD:  Comprehensive, uses 100 references for statistical rigor")


def main():
    """
    Run both pipelines side-by-side for comparison.
    """
    
    print("\n" + "="*70)
    print("DUAL-VALIDATION FRAMEWORK")
    print("Comparing SemanticJudge vs. Sentence-Level Manifold Approach")
    print("="*70)
    
    # Run Pipeline 1: SemanticJudge
    judge_result = pipeline_semantic_judge()
    
    # Run Pipeline 2: Sentence-Level Manifold
    manifold_result = pipeline_sentence_alignment()
    
    # Print comparison
    comparison_matrix(judge_result, manifold_result)
    
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    both_passed = judge_result['verdict']['passed'] and manifold_result.get('validation_results', {}).get('passed', False)
    only_judge = judge_result['verdict']['passed'] and not manifold_result.get('validation_results', {}).get('passed', False)
    only_manifold = not judge_result['verdict']['passed'] and manifold_result.get('validation_results', {}).get('passed', False)
    neither_passed = not judge_result['verdict']['passed'] and not manifold_result.get('validation_results', {}).get('passed', False)
    
    if both_passed:
        print("\n✅ HIGH CONFIDENCE: Both methods agree → ACCEPT")
        confidence = "HIGH"
    elif only_judge:
        print("\n⚠️ MODERATE CONFIDENCE: SemanticJudge passes, manifold questions it")
        confidence = "MEDIUM"
    elif only_manifold:
        print("\n⚠️ MODERATE CONFIDENCE: Manifold+MMD passes, judge questions it")
        confidence = "MEDIUM"
    else:
        print("\n❌ LOW CONFIDENCE: Both methods reject → REJECT")
        confidence = "LOW"
    
    print(f"Confidence Level: {confidence}")
    
    # Save combined results
    combined_results = {
        'semantic_judge_result': judge_result,
        'manifold_result': manifold_result,
        'combined_confidence': confidence,
        'timestamp': str(torch.cuda.Event(enable_timing=True))
    }
    
    print("\n✅ Dual-validation analysis complete!")
    print("   Results saved for dataset curation\n")
    
    return combined_results


if __name__ == "__main__":
    import sys
    
    print("\n⚠️  WARNING: This script loads two large models (~30GB total)")
    print("   Requires: GPU with sufficient VRAM")
    print("   First run may take 5-10 minutes for model downloads\n")
    
    try:
        results = main()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n❌ ERROR: Not enough GPU memory")
            print("   Solution: Use smaller models or quantization")
            print("   See: https://huggingface.co/docs/transformers/quantization")
        else:
            raise
