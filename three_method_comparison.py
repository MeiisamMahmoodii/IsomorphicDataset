"""
Three-Method Alignment Comparison Framework

Compares three approaches to cross-model semantic alignment:

1. KEYWORD-LEVEL (Baseline)
   - Uses 12 pre-selected anchor words
   - Fast but limited scope
   
2. GENERATED SENTENCES (Production)
   - Both models generate responses to 100 identical prompts
   - More realistic manifold alignment
   - Captures model behavior patterns
   
3. REFERENCE MODEL JUDGE (Validation)
   - Llama evaluates semantic distance between generated texts
   - Intra-model consistency measurement
   - Statistical validation approach

This script runs all three in sequence and produces detailed reports.
"""

import torch
import json
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

from source.generator import ConceptGenerator, generate_raw_response
from source.alignment_utils import calculate_procrustes_rotation, apply_alignment_batch
from source.semantic_judge import SemanticJudge
import torch.nn.functional as F

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    "llama": "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
    "mistral": "evolveon/Mistral-7B-Instruct-v0.3-abliterated"
}

# Pre-made anchor words (100) for Method 1
ANCHOR_WORDS = [
    # Science & Technology (10)
    "technology", "innovation", "algorithm", "artificial intelligence", "quantum",
    "computation", "data", "network", "software", "hardware",
    
    # Health & Medicine (10)
    "healthcare", "medicine", "disease", "vaccine", "treatment",
    "prevention", "wellness", "immunity", "diagnosis", "therapy",
    
    # Environment & Sustainability (10)
    "sustainability", "climate", "renewable", "carbon", "ecosystem",
    "conservation", "biodiversity", "pollution", "energy", "nature",
    
    # Education (10)
    "education", "learning", "student", "teacher", "knowledge",
    "skill", "curriculum", "development", "literacy", "training",
    
    # Economics & Business (10)
    "economy", "business", "investment", "trade", "market",
    "profit", "employment", "entrepreneurship", "finance", "commerce",
    
    # Society & Culture (10)
    "society", "culture", "tradition", "community", "relationship",
    "social", "family", "identity", "heritage", "diversity",
    
    # Governance & Ethics (10)
    "governance", "law", "ethics", "justice", "rights",
    "democracy", "institution", "policy", "regulation", "authority",
    
    # Arts & Creativity (10)
    "art", "music", "creativity", "design", "expression",
    "literature", "performance", "imagination", "aesthetic", "beauty",
    
    # Infrastructure & Development (10)
    "infrastructure", "development", "construction", "transportation", "urban",
    "architecture", "engineering", "planning", "facility", "system",
    
    # Psychology & Behavior (10)
    "psychology", "behavior", "emotion", "cognition", "personality",
    "motivation", "resilience", "perception", "learning", "consciousness"
]

# Diverse prompts for Method 2 (100 prompts)
GENERATION_PROMPTS = [
    # Science & Technology (10)
    "Explain the importance of artificial intelligence in modern society",
    "What is the role of quantum computing in the future?",
    "How does renewable energy contribute to sustainability?",
    "Describe the impact of robotics on industry",
    "What are the benefits of biotechnology?",
    "Explain machine learning algorithms",
    "How does blockchain technology work?",
    "What is the significance of nanotechnology?",
    "Describe the evolution of computing",
    "How can technology solve climate change?",
    
    # Health & Wellness (10)
    "What is the importance of preventive healthcare?",
    "How does mental health affect physical wellbeing?",
    "Explain the role of nutrition in disease prevention",
    "What are the benefits of exercise?",
    "How does sleep impact cognitive function?",
    "Describe modern approaches to medicine",
    "What is personalized healthcare?",
    "Explain the immune system's role",
    "How can we reduce healthcare disparities?",
    "What is the future of medical technology?",
    
    # Education (10)
    "What makes an effective educational system?",
    "How has technology transformed learning?",
    "What is the value of critical thinking skills?",
    "Describe the role of teachers in student development",
    "How can education address social inequality?",
    "What is lifelong learning?",
    "Explain the importance of STEM education",
    "How does culture influence education?",
    "What are the benefits of collaborative learning?",
    "How can schools prepare students for the future?",
    
    # Environment & Sustainability (10)
    "What are the causes of climate change?",
    "How can individuals reduce their carbon footprint?",
    "Describe sustainable agriculture practices",
    "What is the importance of biodiversity?",
    "How do renewable resources help sustainability?",
    "Explain the water cycle and its importance",
    "What are the benefits of green spaces?",
    "How can cities become more sustainable?",
    "Describe the impact of deforestation",
    "What is circular economy?",
    
    # Economics & Work (10)
    "What is the gig economy and its impacts?",
    "How does automation affect employment?",
    "Describe the importance of economic diversity",
    "What are the benefits of entrepreneurship?",
    "How can businesses become more ethical?",
    "Explain the role of fair trade",
    "What is financial literacy?",
    "How does globalization affect local economies?",
    "Describe the future of work",
    "What is the value of cooperative businesses?",
    
    # Society & Culture (10)
    "What is the importance of cultural diversity?",
    "How does art contribute to society?",
    "Describe the role of community in wellbeing",
    "What makes communities resilient?",
    "How can we build inclusive societies?",
    "Explain the importance of social solidarity",
    "What is cultural heritage and why preserve it?",
    "How does music influence emotions?",
    "Describe the role of traditions",
    "What is social cohesion?",
    
    # Governance & Ethics (10)
    "What is the importance of transparency in government?",
    "How can we ensure democratic participation?",
    "Describe ethical leadership",
    "What is corporate social responsibility?",
    "How can institutions combat corruption?",
    "Explain the role of law and justice",
    "What is human rights and their importance?",
    "How can communities ensure equitable access?",
    "Describe the role of civil society",
    "What is the social contract?",
    
    # Psychology & Behavior (10)
    "What is emotional intelligence?",
    "How do humans form beliefs?",
    "Describe the nature of motivation",
    "What is the importance of self-awareness?",
    "How do social dynamics affect behavior?",
    "Explain cognitive biases",
    "What is resilience and how to develop it?",
    "How does culture affect psychology?",
    "Describe the role of empathy",
    "What is the nature of human connection?",
    
    # Innovation & Creativity (10)
    "What drives innovation?",
    "How can organizations foster creativity?",
    "Describe the innovation process",
    "What is design thinking?",
    "How does collaboration enhance innovation?",
    "Explain the role of failure in innovation",
    "What is disruptive innovation?",
    "How can individuals become more creative?",
    "Describe the future of innovation",
    "What is the connection between art and innovation?",
    
    # Future & Progress (10)
    "What are the challenges of the next decade?",
    "How can technology improve quality of life?",
    "Describe the role of education in progress",
    "What is sustainable development?",
    "How can we ensure intergenerational equity?",
    "Explain the importance of foresight",
    "What is the role of international cooperation?",
    "How can society manage rapid change?",
    "Describe the future of human civilization",
    "What gives you hope for the future?"
]


# =============================================================================
# METHOD 1: KEYWORD-LEVEL ALIGNMENT
# =============================================================================

def method1_keyword_alignment(gen_llama: ConceptGenerator, gen_mistral: ConceptGenerator) -> Dict:
    """
    Method 1: Extract vectors for pre-defined anchor words.
    Now using 100 diverse keywords across 10 semantic domains.
    """
    print("\n" + "="*80)
    print("METHOD 1: KEYWORD-LEVEL ALIGNMENT (100 Anchor Words)")
    print("="*80)
    
    print("\n📋 Extracting keyword vectors...")
    print(f"Keywords: {ANCHOR_WORDS}\n")
    
    # Extract vectors for each keyword
    llama_keyword_vectors = []
    mistral_keyword_vectors = []
    
    for i, word in enumerate(ANCHOR_WORDS, 1):
        # Llama
        llama_vec = gen_llama.get_latent_vector(word)
        llama_keyword_vectors.append(llama_vec)
        
        # Mistral
        mistral_vec = gen_mistral.get_latent_vector(word)
        mistral_keyword_vectors.append(mistral_vec)
        
        if i % 25 == 0:
            print(f"  ✓ Processed {i}/{len(ANCHOR_WORDS)} keywords")
    
    # Stack into manifolds
    llama_manifold = torch.stack(llama_keyword_vectors)  # [100, 8192]
    mistral_manifold = torch.stack(mistral_keyword_vectors)  # [100, 8192]
    
    print(f"\n✅ Manifold shapes:")
    print(f"   Llama:   {llama_manifold.shape}")
    print(f"   Mistral: {mistral_manifold.shape}")
    
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Compute Procrustes Alignment")
    print("-"*80)
    
    Q, source_mean, target_mean = calculate_procrustes_rotation(
        mistral_manifold, llama_manifold
    )
    
    # Calculate quality metrics
    aligned_mistral = apply_alignment_batch(mistral_manifold, Q, source_mean, target_mean)
    cosine_sims = F.cosine_similarity(aligned_mistral, llama_manifold)
    alignment_quality = cosine_sims.mean().item()
    
    print(f"\n✅ Alignment Quality (mean cosine): {alignment_quality:.4f}")
    print(f"   Top 10 Best Aligned Keywords:")
    sorted_indices = torch.argsort(cosine_sims, descending=True)
    for rank, idx in enumerate(sorted_indices[:10], 1):
        word = ANCHOR_WORDS[idx]
        sim = cosine_sims[idx].item()
        print(f"     {rank:2d}. {word:20s} → {sim:.4f} ✅")
    
    print(f"\n   Top 10 Worst Aligned Keywords:")
    for rank, idx in enumerate(sorted_indices[-10:], 1):
        word = ANCHOR_WORDS[idx]
        sim = cosine_sims[idx].item()
        print(f"     {rank:2d}. {word:20s} → {sim:.4f} ⚠️")
    
    results = {
        "method": "keyword_level",
        "n_anchors": len(ANCHOR_WORDS),
        "llama_manifold": llama_manifold,
        "mistral_manifold": mistral_manifold,
        "Q": Q,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "alignment_quality": alignment_quality,
        "per_anchor_quality": cosine_sims.cpu().numpy().tolist(),
        "std_dev": cosine_sims.std().item(),
        "min_quality": cosine_sims.min().item(),
        "max_quality": cosine_sims.max().item(),
    }
    
    print(f"\n📊 Statistics:")
    print(f"   Mean:   {alignment_quality:.4f}")
    print(f"   Std:    {results['std_dev']:.4f}")
    print(f"   Min:    {results['min_quality']:.4f}")
    print(f"   Max:    {results['max_quality']:.4f}")
    
    return results


# =============================================================================
# METHOD 2: GENERATED SENTENCES ALIGNMENT
# =============================================================================

def method2_generated_sentences_alignment(gen_llama: ConceptGenerator, gen_mistral: ConceptGenerator) -> Dict:
    """
    Method 2: Generate responses to diverse prompts, then align manifolds.
    More realistic - captures natural model behavior.
    """
    print("\n" + "="*80)
    print("METHOD 2: GENERATED SENTENCES ALIGNMENT (100 Diverse Prompts)")
    print("="*80)
    
    print("\n🔄 Generating sentences from identical prompts...")
    print(f"Using {len(GENERATION_PROMPTS)} diverse prompts\n")
    
    llama_texts = []
    mistral_texts = []
    
    # Generate from both models on same prompts
    for i, prompt in enumerate(GENERATION_PROMPTS, 1):
        # Generate from Llama
        llama_response = generate_raw_response(prompt, gen_llama.model, gen_llama.tokenizer)
        llama_texts.append(llama_response)
        
        # Generate from Mistral
        mistral_response = generate_raw_response(prompt, gen_mistral.model, gen_mistral.tokenizer)
        mistral_texts.append(mistral_response)
        
        if i % 20 == 0:
            print(f"  ✓ Generated {i}/{len(GENERATION_PROMPTS)} prompt pairs")
    
    print(f"\n✅ Generated {len(llama_texts)} text pairs")
    print(f"   Sample Llama response:   {llama_texts[0][:60]}...")
    print(f"   Sample Mistral response: {mistral_texts[0][:60]}...")
    
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Extract Sentence Vectors")
    print("-"*80)
    
    print("\nExtracting vectors from Llama responses...")
    llama_vectors = []
    for i in range(0, len(llama_texts), 20):
        batch = llama_texts[i:i+20]
        for text in batch:
            vec = gen_llama.get_latent_vector(text)
            llama_vectors.append(vec)
        print(f"  ✓ Processed {min(i+20, len(llama_texts))}/{len(llama_texts)} texts")
    
    print("\nExtracting vectors from Mistral responses...")
    mistral_vectors = []
    for i in range(0, len(mistral_texts), 20):
        batch = mistral_texts[i:i+20]
        for text in batch:
            vec = gen_mistral.get_latent_vector(text)
            mistral_vectors.append(vec)
        print(f"  ✓ Processed {min(i+20, len(mistral_texts))}/{len(mistral_texts)} texts")
    
    llama_manifold = torch.stack(llama_vectors)  # [100, 8192]
    mistral_manifold = torch.stack(mistral_vectors)  # [100, 8192]
    
    print(f"\n✅ Manifold shapes:")
    print(f"   Llama:   {llama_manifold.shape}")
    print(f"   Mistral: {mistral_manifold.shape}")
    
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Compute Procrustes Alignment")
    print("-"*80)
    
    Q, source_mean, target_mean = calculate_procrustes_rotation(
        mistral_manifold, llama_manifold
    )
    
    # Calculate quality metrics
    aligned_mistral = apply_alignment_batch(mistral_manifold, Q, source_mean, target_mean)
    cosine_sims = F.cosine_similarity(aligned_mistral, llama_manifold)
    alignment_quality = cosine_sims.mean().item()
    
    print(f"\n✅ Alignment Quality (mean cosine): {alignment_quality:.4f}")
    print(f"   Distribution of per-sentence similarities:")
    
    sims_np = cosine_sims.cpu().numpy()
    percentiles = [0, 25, 50, 75, 100]
    for p in percentiles:
        val = np.percentile(sims_np, p)
        print(f"     {p:3d}th percentile: {val:.4f}")
    
    results = {
        "method": "generated_sentences",
        "n_sentences": len(llama_texts),
        "llama_manifold": llama_manifold,
        "mistral_manifold": mistral_manifold,
        "llama_texts": llama_texts,
        "mistral_texts": mistral_texts,
        "Q": Q,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "alignment_quality": alignment_quality,
        "per_sentence_quality": cosine_sims.cpu().numpy().tolist(),
        "std_dev": cosine_sims.std().item(),
        "min_quality": cosine_sims.min().item(),
        "max_quality": cosine_sims.max().item(),
    }
    
    print(f"\n📊 Statistics:")
    print(f"   Mean:   {alignment_quality:.4f}")
    print(f"   Std:    {results['std_dev']:.4f}")
    print(f"   Min:    {results['min_quality']:.4f}")
    print(f"   Max:    {results['max_quality']:.4f}")
    
    return results


# =============================================================================
# METHOD 3: REFERENCE MODEL JUDGE
# =============================================================================

def method3_reference_model_judge(
    gen_llama: ConceptGenerator,
    gen_mistral: ConceptGenerator,
    method2_results: Dict
) -> Dict:
    """
    Method 3: Use Llama as judge to measure semantic distance
    between generated texts. Intra-model consistency.
    """
    print("\n" + "="*80)
    print("METHOD 3: REFERENCE MODEL JUDGE (Llama as Authority)")
    print("="*80)
    
    print("\n⚖️ Initializing Llama as semantic judge...")
    judge = SemanticJudge(gen_llama.model, gen_llama.tokenizer)
    
    llama_texts = method2_results["llama_texts"]
    mistral_texts = method2_results["mistral_texts"]
    
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Evaluate Semantic Distance Between Generated Texts")
    print("-"*80)
    
    print(f"\nEvaluating {len(llama_texts)} text pairs in Llama's coordinate system")
    print("(Does Llama think both texts say the same thing?)\n")
    
    metrics_list = []
    verdicts_list = []
    
    for i, (llama_text, mistral_text) in enumerate(zip(llama_texts, mistral_texts), 1):
        metrics = judge.evaluate_isomorphism(llama_text, mistral_text, use_wasserstein=False)
        verdict = judge.apply_thresholds(metrics, cosine_threshold=0.85, distance_threshold=0.05)
        
        metrics_list.append(metrics)
        verdicts_list.append(verdict['passed'])
        
        if i % 25 == 0:
            print(f"  ✓ Evaluated {i}/{len(llama_texts)} pairs")
    
    print(f"\n✅ Evaluated all {len(llama_texts)} text pairs")
    
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Analyze Judge's Verdicts")
    print("-"*80)
    
    cosine_sims = [m['cosine_similarity'] for m in metrics_list]
    euclidean_dists = [m['euclidean_distance'] for m in metrics_list]
    
    passed_count = sum(verdicts_list)
    pass_rate = passed_count / len(verdicts_list) * 100
    
    print(f"\n✅ Judge's Verdict Summary:")
    print(f"   Passed (≥0.85 cosine):  {passed_count}/{len(verdicts_list)} ({pass_rate:.1f}%)")
    print(f"   Failed (<0.85 cosine):  {len(verdicts_list)-passed_count}/{len(verdicts_list)} ({100-pass_rate:.1f}%)")
    
    print(f"\n📊 Cosine Similarity Statistics (Judge's View):")
    print(f"   Mean:   {np.mean(cosine_sims):.4f}")
    print(f"   Std:    {np.std(cosine_sims):.4f}")
    print(f"   Min:    {np.min(cosine_sims):.4f}")
    print(f"   Max:    {np.max(cosine_sims):.4f}")
    
    print(f"\n📊 Euclidean Distance Statistics:")
    print(f"   Mean:   {np.mean(euclidean_dists):.4f}")
    print(f"   Std:    {np.std(euclidean_dists):.4f}")
    print(f"   Min:    {np.min(euclidean_dists):.4f}")
    print(f"   Max:    {np.max(euclidean_dists):.4f}")
    
    # Find best and worst pairs
    best_idx = np.argmax(cosine_sims)
    worst_idx = np.argmin(cosine_sims)
    
    print(f"\n🏆 Best Alignment (Cosine={cosine_sims[best_idx]:.4f}):")
    print(f"   Llama:   {llama_texts[best_idx][:70]}...")
    print(f"   Mistral: {mistral_texts[best_idx][:70]}...")
    
    print(f"\n⚠️ Worst Alignment (Cosine={cosine_sims[worst_idx]:.4f}):")
    print(f"   Llama:   {llama_texts[worst_idx][:70]}...")
    print(f"   Mistral: {mistral_texts[worst_idx][:70]}...")
    
    results = {
        "method": "reference_model_judge",
        "judge_model": "Llama",
        "n_pairs_evaluated": len(llama_texts),
        "passed_verdicts": passed_count,
        "pass_rate": pass_rate,
        "cosine_similarities": cosine_sims,
        "euclidean_distances": euclidean_dists,
        "mean_cosine": np.mean(cosine_sims),
        "std_cosine": np.std(cosine_sims),
        "mean_euclidean": np.mean(euclidean_dists),
        "std_euclidean": np.std(euclidean_dists),
    }
    
    return results


# =============================================================================
# COMPARISON & REPORTING
# =============================================================================

def generate_comparison_report(results1: Dict, results2: Dict, results3: Dict):
    """Generate comprehensive comparison report of all three methods."""
    
    print("\n\n" + "="*80)
    print("COMPREHENSIVE COMPARISON REPORT")
    print("="*80)
    
    # ========================================================================
    print("\n" + "="*80)
    print("1. ALIGNMENT QUALITY COMPARISON")
    print("="*80)
    
    print(f"\n{'Method':<30} {'Quality':<15} {'Std Dev':<15} {'Range':<30}")
    print("-" * 80)
    
    print(f"{'Keyword-Level (12)':<30} {results1['alignment_quality']:<15.4f} {results1['std_dev']:<15.4f} [{results1['min_quality']:.4f}, {results1['max_quality']:.4f}]")
    print(f"{'Generated Sentences (100)':<30} {results2['alignment_quality']:<15.4f} {results2['std_dev']:<15.4f} [{results2['min_quality']:.4f}, {results2['max_quality']:.4f}]")
    print(f"{'Reference Judge (Cosine)':<30} {results3['mean_cosine']:<15.4f} {results3['std_cosine']:<15.4f} [{min(results3['cosine_similarities']):.4f}, {max(results3['cosine_similarities']):.4f}]")
    
    # ========================================================================
    print("\n" + "="*80)
    print("2. CONSISTENCY ANALYSIS")
    print("="*80)
    
    print(f"\nMethod 1 (Keywords):")
    print(f"  - Uses: {results1['n_anchors']} pre-defined anchor words")
    print(f"  - Consistency: {'High' if results1['std_dev'] < 0.1 else 'Medium' if results1['std_dev'] < 0.15 else 'Low'} (Std={results1['std_dev']:.4f})")
    print(f"  - Coverage: Limited to dictionary words only")
    
    print(f"\nMethod 2 (Generated Sentences):")
    print(f"  - Uses: {results2['n_sentences']} generated text pairs")
    print(f"  - Consistency: {'High' if results2['std_dev'] < 0.1 else 'Medium' if results2['std_dev'] < 0.15 else 'Low'} (Std={results2['std_dev']:.4f})")
    print(f"  - Coverage: Broad (diverse topics, natural generation patterns)")
    
    print(f"\nMethod 3 (Reference Judge):")
    print(f"  - Evaluates: {results3['n_pairs_evaluated']} generated pairs")
    print(f"  - Pass Rate: {results3['pass_rate']:.1f}% (threshold: cosine > 0.85)")
    print(f"  - Consistency: {'High' if results3['std_cosine'] < 0.1 else 'Medium' if results3['std_cosine'] < 0.15 else 'Low'} (Std={results3['std_cosine']:.4f})")
    
    # ========================================================================
    print("\n" + "="*80)
    print("3. STRENGTHS & WEAKNESSES")
    print("="*80)
    
    method1_quality = results1['alignment_quality']
    method2_quality = results2['alignment_quality']
    method3_pass = results3['pass_rate']
    
    print(f"\n📊 Method 1 - Keyword-Level:")
    print(f"   Strengths:")
    print(f"     ✅ Fast (only 12 vectors)")
    print(f"     ✅ Reproducible (fixed keywords)")
    print(f"     ✅ Simple to interpret")
    print(f"   Weaknesses:")
    print(f"     ❌ Limited scope (only 12 points)")
    print(f"     ❌ Doesn't capture generation patterns")
    print(f"     ❌ May miss important semantic dimensions")
    print(f"   Quality Score: {method1_quality:.4f}")
    
    print(f"\n📊 Method 2 - Generated Sentences:")
    print(f"   Strengths:")
    print(f"     ✅ Captures real model behavior")
    print(f"     ✅ Diverse (100+ topics)")
    print(f"     ✅ Manifold is more comprehensive")
    print(f"   Weaknesses:")
    print(f"     ❌ Slower (generates 200 texts)")
    print(f"     ❌ Variation across runs")
    print(f"     ❌ Generation can introduce noise")
    print(f"   Quality Score: {method2_quality:.4f} {'👑' if method2_quality > method1_quality else ''}")
    
    print(f"\n📊 Method 3 - Reference Judge:")
    print(f"   Strengths:")
    print(f"     ✅ Independent validation")
    print(f"     ✅ Measures actual semantic equivalence")
    print(f"     ✅ Provides high-confidence filtering")
    print(f"   Weaknesses:")
    print(f"     ❌ Requires running texts through judge")
    print(f"     ❌ Judge model dependent (Llama opinion)")
    print(f"     ❌ Doesn't directly measure alignment")
    print(f"   Pass Rate: {method3_pass:.1f}% {'👑' if method3_pass > 80 else ''}")
    
    # ========================================================================
    print("\n" + "="*80)
    print("4. RECOMMENDED WORKFLOW")
    print("="*80)
    
    print(f"\n✅ RECOMMENDED APPROACH:")
    print(f"   1. Start with Method 2 (Generated Sentences)")
    print(f"      → Captures real model behavior")
    print(f"      → Quality: {method2_quality:.4f}")
    print(f"\n   2. Validate with Method 3 (Reference Judge)")
    print(f"      → Filter high-confidence pairs")
    print(f"      → Pass Rate: {method3_pass:.1f}%")
    print(f"\n   3. Use Method 1 for quick testing")
    print(f"      → Fast iteration (12 keywords)")
    print(f"      → Lightweight baseline")
    
    # ========================================================================
    print("\n" + "="*80)
    print("5. KEY FINDINGS")
    print("="*80)
    
    if method2_quality > method1_quality + 0.05:
        print(f"\n🔍 Finding 1: Generated sentences provide better alignment")
        print(f"   Method 2 is {(method2_quality - method1_quality)/method1_quality * 100:.1f}% better than Method 1")
    
    if method3_pass > 70:
        print(f"\n🔍 Finding 2: Reference judge confirms high semantic agreement")
        print(f"   {method3_pass:.1f}% of generated pairs are semantically equivalent (per Llama)")
    
    if results2['std_dev'] < results1['std_dev']:
        print(f"\n🔍 Finding 3: Generated sentences show more consistent alignment")
        print(f"   Method 2 variance (std={results2['std_dev']:.4f}) < Method 1 (std={results1['std_dev']:.4f})")
    
    consistency_gap = abs(method2_quality - method3_pass/100)
    if consistency_gap > 0.1:
        print(f"\n🔍 Finding 4: Procrustes alignment vs Reference judge differ")
        print(f"   Gap: {consistency_gap:.4f}")
        print(f"   → Suggests models align mathematically but may diverge semantically")
    
    # ========================================================================
    print("\n" + "="*80)
    print("6. STATISTICAL SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<40} {'Method 1':<15} {'Method 2':<15} {'Method 3':<15}")
    print("-" * 85)
    print(f"{'Alignment/Pass Quality':<40} {method1_quality:<15.4f} {method2_quality:<15.4f} {method3_pass/100:<15.4f}")
    print(f"{'Consistency (Std Dev)':<40} {results1['std_dev']:<15.4f} {results2['std_dev']:<15.4f} {results3['std_cosine']:<15.4f}")
    print(f"{'Sample Count':<40} {results1['n_anchors']:<15} {results2['n_sentences']:<15} {results3['n_pairs_evaluated']:<15}")
    print(f"{'Coverage':<40} {'Limited':<15} {'Comprehensive':<15} {'Full':<15}")


def main():
    """Main execution: Run all three methods and generate reports."""
    
    print("\n" + "="*80)
    print("THREE-METHOD ALIGNMENT COMPARISON FRAMEWORK")
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
    # Run all three methods
    try:
        # Method 1: Keywords
        results1 = method1_keyword_alignment(gen_llama, gen_mistral)
        
        # Method 2: Generated Sentences
        results2 = method2_generated_sentences_alignment(gen_llama, gen_mistral)
        
        # Method 3: Reference Judge
        results3 = method3_reference_model_judge(gen_llama, gen_mistral, results2)
        
        # Generate comparison report
        generate_comparison_report(results1, results2, results3)
        
        # ====================================================================
        print("\n" + "="*80)
        print("FINAL RECOMMENDATION")
        print("="*80)
        
        best_method = max([
            ("Keyword-Level", results1['alignment_quality']),
            ("Generated Sentences", results2['alignment_quality']),
            ("Reference Judge Pass Rate", results3['pass_rate']/100)
        ], key=lambda x: x[1])
        
        print(f"\n🏆 Best Performing Method: {best_method[0]}")
        print(f"   Score: {best_method[1]:.4f}")
        
        print(f"\n✅ For production use:")
        print(f"   → Use Method 2 (Generated Sentences) for alignment")
        print(f"   → Validate with Method 3 (Reference Judge) for filtering")
        print(f"   → Monitor with Method 1 (Keywords) for quick checks")
        
        print(f"\n📊 Combined Pipeline Success Rate:")
        combined_success = results2['alignment_quality'] * (results3['pass_rate']/100)
        print(f"   {results2['alignment_quality']:.4f} × {results3['pass_rate']/100:.4f} = {combined_success:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        raise
    
    print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
