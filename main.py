import torch
import torch.nn.functional as F
from source.generator import ConceptGenerator
from source.alignment import (
    extract_anchor_vectors,
    align_latent_spaces,
    apply_rotation_to_vectors,
    compute_alignment_metrics,
    get_all_anchor_words
)
from source.alignment_utils import apply_alignment


def main():
    # Define models to compare
    # Original models:
    # model_names = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
    
    # Abliterated models (without refusal filters):
    model_names = ["failspy/Meta-Llama-3-8B-Instruct-abliterated-v3", "evolveon/Mistral-7B-Instruct-v0.3-abliterated"]
    
    # Define your Seed and Forbidden Concepts
    seed = "the only thing that can be done to improve the quality of life in the inner city is to build a giant wall to keep the blacks out"
    forbidden = ["blacks", "wall", "city"]
    
    # Enable perspective injection for more semantic alignment
    use_perspective_injection = True
    
    # Store results for each model
    results = {}
    
    print(f"\n{'='*60}")
    print("Perspective Injection: ENABLED ✓")
    print("(Models will maintain consistent perspective for true isomorphism)")
    print(f"{'='*60}")
    
    # Generate variations and vectors for each model
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*60}")
        
        gen = ConceptGenerator(model_name)
        
        # 3. Call the generation for different lengths
        print("\n--- Generating Short Variation (5-10 words) ---")
        valid_short = gen.get_validated_variation(
            seed, forbidden, 5, 10, 
            maintain_perspective=use_perspective_injection
        )
        
        print("\n--- Generating Long Variation (15-20 words) ---")
        valid_long = gen.get_validated_variation(
            seed, forbidden, 15, 20,
            maintain_perspective=use_perspective_injection
        )
        
        # Output results
        print(f"\nShort (5-10 words): {valid_short}")
        print(f"Long (15-20 words): {valid_long}")
        
        # ===== METHOD 1: MEAN POOLING =====
        print("\n--- METHOD 1: ATTENTION-MASKED MEAN POOLING ---")
        vector_short_mean = gen.get_latent_vector(valid_short)
        vector_long_mean = gen.get_latent_vector(valid_long)
        
        similarity_mean = F.cosine_similarity(vector_short_mean.unsqueeze(0), vector_long_mean.unsqueeze(0))
        print(f"Intra-Model Concept Stability (Mean Pooling): {similarity_mean.item():.4f}")
        
        # ===== METHOD 2: LAST TOKEN ONLY =====
        print("\n--- METHOD 2: LAST TOKEN ONLY ---")
        vector_short_last = gen.get_last_token_vector(valid_short)
        vector_long_last = gen.get_last_token_vector(valid_long)
        
        similarity_last = F.cosine_similarity(vector_short_last.unsqueeze(0), vector_long_last.unsqueeze(0))
        print(f"Intra-Model Concept Stability (Last Token): {similarity_last.item():.4f}")
        
        # ===== METHOD 3: HYBRID (Mean Pooling + Last Token) =====
        print("\n--- METHOD 3: HYBRID (Mean Pooling + Last Token) ---")
        vector_short_hybrid = gen.get_hybrid_vector(valid_short)
        vector_long_hybrid = gen.get_hybrid_vector(valid_long)
        
        similarity_hybrid = F.cosine_similarity(vector_short_hybrid.unsqueeze(0), vector_long_hybrid.unsqueeze(0))
        print(f"Intra-Model Concept Stability (Hybrid): {similarity_hybrid.item():.4f}")
        
        # Store results for cross-model analysis
        results[model_name] = {
            'valid_short': valid_short,
            'valid_long': valid_long,
            'vector_short_mean': vector_short_mean,
            'vector_long_mean': vector_long_mean,
            'vector_short_last': vector_short_last,
            'vector_long_last': vector_long_last,
            'vector_short_hybrid': vector_short_hybrid,
            'vector_long_hybrid': vector_long_hybrid,
            'intra_similarity_mean': similarity_mean.item(),
            'intra_similarity_last': similarity_last.item(),
            'intra_similarity_hybrid': similarity_hybrid.item()
        }
    
    # Cross-Model Alignment Analysis
    print(f"\n{'='*60}")
    print("CROSS-MODEL ALIGNMENT ANALYSIS")
    print(f"{'='*60}")
    
    model_list = list(results.keys())
    model_1 = model_list[0]
    model_2 = model_list[1]
    
    # ===== METHOD 1: MEAN POOLING =====
    print("\n--- METHOD 1: MEAN POOLING CROSS-MODEL ---")
    cross_sim_short_mean = F.cosine_similarity(
        results[model_1]['vector_short_mean'].unsqueeze(0), 
        results[model_2]['vector_short_mean'].unsqueeze(0)
    )
    cross_sim_long_mean = F.cosine_similarity(
        results[model_1]['vector_long_mean'].unsqueeze(0), 
        results[model_2]['vector_long_mean'].unsqueeze(0)
    )
    
    print(f"Cross-Model Similarity (Short variants - Mean): {cross_sim_short_mean.item():.4f}")
    print(f"Cross-Model Similarity (Long variants - Mean): {cross_sim_long_mean.item():.4f}")
    
    # ===== METHOD 2: LAST TOKEN ONLY =====
    print("\n--- METHOD 2: LAST TOKEN CROSS-MODEL ---")
    cross_sim_short_last = F.cosine_similarity(
        results[model_1]['vector_short_last'].unsqueeze(0), 
        results[model_2]['vector_short_last'].unsqueeze(0)
    )
    cross_sim_long_last = F.cosine_similarity(
        results[model_1]['vector_long_last'].unsqueeze(0), 
        results[model_2]['vector_long_last'].unsqueeze(0)
    )
    
    print(f"Cross-Model Similarity (Short variants - Last Token): {cross_sim_short_last.item():.4f}")
    print(f"Cross-Model Similarity (Long variants - Last Token): {cross_sim_long_last.item():.4f}")
    
    # ===== METHOD 3: HYBRID (Mean Pooling + Last Token) =====
    print("\n--- METHOD 3: HYBRID CROSS-MODEL ---")
    cross_sim_short_hybrid = F.cosine_similarity(
        results[model_1]['vector_short_hybrid'].unsqueeze(0), 
        results[model_2]['vector_short_hybrid'].unsqueeze(0)
    )
    cross_sim_long_hybrid = F.cosine_similarity(
        results[model_1]['vector_long_hybrid'].unsqueeze(0), 
        results[model_2]['vector_long_hybrid'].unsqueeze(0)
    )
    
    print(f"Cross-Model Similarity (Short variants - Hybrid): {cross_sim_short_hybrid.item():.4f}")
    print(f"Cross-Model Similarity (Long variants - Hybrid): {cross_sim_long_hybrid.item():.4f}")
    
    # Summary Table
    print(f"\n{'='*60}")
    print("SUMMARY: METHOD COMPARISON")
    print(f"{'='*60}")
    for model_name, data in results.items():
        print(f"\n{model_name}")
        print(f"  Intra-Model Stability (Mean Pooling): {data['intra_similarity_mean']:.4f}")
        print(f"  Intra-Model Stability (Last Token): {data['intra_similarity_last']:.4f}")
        print(f"  Intra-Model Stability (Hybrid): {data['intra_similarity_hybrid']:.4f}")
        print(f"  Short Variation: {data['valid_short']}")
        print(f"  Long Variation: {data['valid_long']}")
    
    # ===== LATENT SPACE ALIGNMENT USING ANCHOR WORDS =====
    print(f"\n{'='*60}")
    print("LATENT SPACE ALIGNMENT (Procrustes with Anchor Words)")
    print(f"{'='*60}")
    
    # Get generators for both models
    gen_model_1 = ConceptGenerator(model_1)
    gen_model_2 = ConceptGenerator(model_2)
    
    # Extract anchor vectors using hybrid method
    print(f"\nModel 1: {model_1}")
    anchors_model_1 = extract_anchor_vectors(gen_model_1, method="hybrid")
    
    print(f"\nModel 2: {model_2}")
    anchors_model_2 = extract_anchor_vectors(gen_model_2, method="hybrid")
    
    # Compute rotation matrix
    print("\nComputing alignment rotation matrix...")
    Q, alignment_quality, source_mean, target_mean = align_latent_spaces(anchors_model_1, anchors_model_2)
    
    # Apply rotation to model 1's concept vectors using proper Procrustes transformation
    print("\nApplying rotation to concept vectors...")
    short_aligned = apply_alignment(
        results[model_1]['vector_short_hybrid'],
        Q,
        source_mean,
        target_mean
    )
    long_aligned = apply_alignment(
        results[model_1]['vector_long_hybrid'],
        Q,
        source_mean,
        target_mean
    )
    
    # Get target vectors
    short_target = results[model_2]['vector_short_hybrid']
    long_target = results[model_2]['vector_long_hybrid']
    
    # Cosine similarity after alignment
    aligned_sim_short = F.cosine_similarity(short_aligned.unsqueeze(0), short_target.unsqueeze(0))
    aligned_sim_long = F.cosine_similarity(long_aligned.unsqueeze(0), long_target.unsqueeze(0))
    
    print(f"\n{'='*60}")
    print("ALIGNMENT RESULTS")
    print(f"{'='*60}")
    print(f"\nAnchor Word Alignment Quality: {alignment_quality:.4f}")
    print(f"  (Based on {len(anchors_model_1)} shared anchor words)")
    
    print(f"\nPost-Alignment Concept Similarity:")
    print(f"  Short Variants: {aligned_sim_short.item():.4f} (before: {cross_sim_short_hybrid.item():.4f})")
    print(f"  Long Variants: {aligned_sim_long.item():.4f} (before: {cross_sim_long_hybrid.item():.4f})")
    
    print(f"\nImprovement:")
    improvement_short = aligned_sim_short.item() - cross_sim_short_hybrid.item()
    improvement_long = aligned_sim_long.item() - cross_sim_long_hybrid.item()
    print(f"  Short Variants: +{improvement_short:.4f}")
    print(f"  Long Variants: +{improvement_long:.4f}")
    
    # Verification of rotation matrix properties
    print(f"\n{'='*60}")
    print("PROCRUSTES ROTATION MATRIX PROPERTIES")
    print(f"{'='*60}")
    is_orthogonal = torch.allclose(Q @ Q.T, torch.eye(Q.shape[0]), atol=1e-4)
    det_Q = torch.det(Q).item()
    print(f"\nRotation Matrix Q: {Q.shape}")
    print(f"  Is orthogonal: {is_orthogonal} ✓" if is_orthogonal else f"  Is orthogonal: {is_orthogonal}")
    print(f"  Determinant: {det_Q:.6f} (±1 = true rotation)")
    
    print(f"\n{'='*60}")
    print("📚 FOR DETAILED STEP-BY-STEP EXAMPLE")
    print(f"{'='*60}")
    print("Run: python example_cross_model_alignment.py")
    print("This shows the complete workflow with explanations.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
