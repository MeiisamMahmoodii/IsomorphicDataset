"""
Sentence Anchors Module

Implements Parallel Sentence Procrustes alignment instead of word-level anchors.
Uses meaningful sentence pairs to align latent manifolds rather than dictionary vectors.

This solves the "compositional geometry" problem: words may align but sentences
expressing complex concepts may still diverge between models.

[cite: Parallel Sentence Alignment, 187-189]
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .alignment_utils import calculate_procrustes_rotation, compute_alignment_quality


# 100 Neutral Reference Sentences for Sentence-Level Anchors
# These are non-toxic, diverse sentences suitable for proof-of-concept
REFERENCE_SENTENCES = [
    # Science & Technology (10)
    "Mathematics is the foundation of physics.",
    "The speed of light is constant in vacuum.",
    "Oxygen is essential for human respiration.",
    "Computers process information using binary code.",
    "Quantum mechanics describes subatomic behavior.",
    "DNA contains genetic instructions for life.",
    "Water consists of hydrogen and oxygen molecules.",
    "Electricity flows through conductive materials.",
    "The Earth orbits around the Sun.",
    "Artificial intelligence mimics human cognition.",
    
    # Nature & Geography (10)
    "Mountains are formed by tectonic plate collision.",
    "Rivers flow from high elevation to low areas.",
    "Forests provide oxygen and absorb carbon dioxide.",
    "Weather patterns are influenced by ocean currents.",
    "Volcanoes form where tectonic plates meet.",
    "Desert ecosystems adapt to extreme dryness.",
    "Tropical rainforests contain the most biodiversity.",
    "Glaciers store frozen water from past millennia.",
    "The ocean covers more than seventy percent of Earth.",
    "Soil fertility depends on mineral and nutrient content.",
    
    # Society & Culture (10)
    "Languages evolve through cultural interaction.",
    "Art reflects the values of its time.",
    "Education is fundamental to social development.",
    "Trade connects communities across distances.",
    "Music transcends cultural and linguistic barriers.",
    "Architecture reflects engineering and aesthetic principles.",
    "Literature preserves stories and cultural memory.",
    "Philosophy examines fundamental questions about existence.",
    "Sports build community and encourage cooperation.",
    "Food traditions connect people to heritage.",
    
    # History & Events (10)
    "The printing press revolutionized information distribution.",
    "The Industrial Revolution transformed manufacturing.",
    "Scientific discoveries often challenge existing beliefs.",
    "Wars have lasting impacts on societies.",
    "Democracy evolved through centuries of political struggle.",
    "Technology adoption follows predictable adoption curves.",
    "Civilizations rise and fall due to various factors.",
    "Trade routes historically connected distant regions.",
    "Exploration expanded geographical knowledge and understanding.",
    "Revolutions often follow periods of social tension.",
    
    # Economics & Work (10)
    "Supply and demand determine market prices.",
    "Labor specialization increases economic efficiency.",
    "Investment in education yields long-term returns.",
    "Inflation reduces purchasing power over time.",
    "Businesses must balance profit and sustainability.",
    "International trade promotes economic growth.",
    "Employment provides income and social status.",
    "Infrastructure development attracts economic activity.",
    "Innovation drives competitive advantage in markets.",
    "Resources are distributed unevenly across regions.",
    
    # Health & Medicine (10)
    "Exercise strengthens cardiovascular and muscle systems.",
    "Vaccination prevents infectious disease outbreaks.",
    "Nutrition affects physical and mental health.",
    "Sleep is essential for cognitive function.",
    "Stress management improves overall well-being.",
    "Regular checkups detect health problems early.",
    "Mental health is as important as physical health.",
    "Antibiotics treat bacterial infections effectively.",
    "Hygiene practices prevent disease transmission.",
    "Healthy habits extend life expectancy.",
    
    # Environment & Sustainability (10)
    "Renewable energy reduces carbon emissions.",
    "Pollution damages ecosystems and human health.",
    "Conservation protects endangered species.",
    "Recycling reduces waste in landfills.",
    "Climate change affects weather patterns globally.",
    "Sustainable farming preserves soil and water.",
    "Biodiversity strengthens ecosystem resilience.",
    "Green spaces improve air quality in cities.",
    "Ocean acidification threatens marine life.",
    "Deforestation contributes to climate change.",
    
    # Psychology & Behavior (10)
    "Habit formation requires consistent repetition.",
    "Motivation drives human effort and achievement.",
    "Memory consolidation occurs during sleep.",
    "Decision-making involves weighing options.",
    "Emotions influence perception and judgment.",
    "Learning occurs through experience and practice.",
    "Personality traits remain relatively stable.",
    "Cooperation benefits groups more than competition.",
    "Empathy enables understanding of others.",
    "Confidence affects performance in challenging tasks.",
    
    # Philosophy & Ethics (10)
    "Morality varies across different cultures.",
    "Ethics guide decision-making in complex situations.",
    "Truth-seeking requires open-mindedness.",
    "Freedom involves responsibility for actions.",
    "Justice requires fair treatment of all people.",
    "Kindness creates positive social interactions.",
    "Honesty builds trust in relationships.",
    "Fairness is important in exchange and cooperation.",
    "Respect acknowledges the dignity of others.",
    "Compassion responds to suffering with care.",
    
    # Communication & Language (10)
    "Clear communication requires shared understanding.",
    "Language shapes how we perceive reality.",
    "Active listening improves relationship quality.",
    "Feedback helps refine and improve work.",
    "Storytelling conveys meaning across generations.",
    "Vocabulary expands through reading and exposure.",
    "Tone affects how messages are received.",
    "Dialogue resolves conflicts better than arguments.",
    "Translation preserves meaning across languages.",
    "Imagery enhances written and spoken descriptions."
]


def get_reference_sentences(num_sentences: int = 100) -> List[str]:
    """
    Get reference sentences for sentence-level anchors.
    
    Args:
        num_sentences (int): Number of sentences (max 100)
        
    Returns:
        List[str]: Reference sentences
    """
    return REFERENCE_SENTENCES[:min(num_sentences, len(REFERENCE_SENTENCES))]


def extract_sentence_vectors(
    generator,
    sentences: List[str],
    method: str = "hybrid"
) -> Dict[str, torch.Tensor]:
    """
    Extract latent vectors for a list of sentences.
    
    Args:
        generator: ConceptGenerator instance
        sentences (List[str]): Sentences to vectorize
        method (str): "mean", "last", or "hybrid"
        
    Returns:
        Dict[str, torch.Tensor]: Sentence -> vector mapping
    """
    vectors = {}
    
    print(f"\nExtracting {len(sentences)} sentence vectors using {method} method...")
    
    for i, sentence in enumerate(sentences):
        try:
            if method == "mean":
                vector = generator.get_latent_vector(sentence)
            elif method == "last":
                vector = generator.get_last_token_vector(sentence)
            elif method == "hybrid":
                vector = generator.get_hybrid_vector(sentence)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            vectors[sentence] = vector
            
            if (i + 1) % 20 == 0:
                print(f"  ✓ Processed {i+1}/{len(sentences)} sentences")
        
        except Exception as e:
            print(f"  ⚠️ Failed for sentence {i}: {e}")
            continue
    
    print(f"✅ Successfully extracted {len(vectors)} sentence vectors")
    return vectors


def compute_sentence_procrustes_alignment(
    sentences_source: Dict[str, torch.Tensor],
    sentences_target: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Compute Procrustes alignment using parallel sentences instead of words.
    
    This aligns the "manifold" (complex shape) of latent space where both
    models encode full compositional thoughts.
    
    Args:
        sentences_source (Dict): Source model sentence vectors
        sentences_target (Dict): Target model sentence vectors
        
    Returns:
        Tuple: (Rotation matrix Q, alignment quality, source_mean, target_mean)
    """
    
    # Find common sentences
    common_sentences = set(sentences_source.keys()) & set(sentences_target.keys())
    
    if len(common_sentences) < 2:
        raise ValueError(f"Not enough common sentences: {len(common_sentences)}")
    
    print(f"\n📊 Aligning using {len(common_sentences)} parallel sentences")
    print("   (These form the 'manifold' of latent space)")
    
    # Stack vectors for common sentences
    X_list = [sentences_source[s] for s in common_sentences]
    Y_list = [sentences_target[s] for s in common_sentences]
    
    X = torch.stack(X_list, dim=0)  # Source manifold
    Y = torch.stack(Y_list, dim=0)  # Target manifold
    
    print(f"\nManifold shapes:")
    print(f"  Source: {X.shape} (sentences × hidden_dim)")
    print(f"  Target: {Y.shape} (sentences × hidden_dim)")
    
    # Solve Procrustes using SVD
    Q, source_mean, target_mean = calculate_procrustes_rotation(X, Y)
    
    # Compute alignment quality
    alignment_quality = compute_alignment_quality(X, Y, Q, source_mean, target_mean)
    
    print(f"\n✅ Manifold Alignment Complete")
    print(f"   Quality (mean cosine similarity): {alignment_quality:.4f}")
    
    return Q, alignment_quality, source_mean, target_mean


def analyze_sentence_alignment(
    sentences: Dict[str, torch.Tensor],
    sentences_target: Dict[str, torch.Tensor],
    Q: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
    top_k: int = 5
) -> Dict:
    """
    Analyze alignment quality on individual sentences.
    
    Args:
        sentences: Source sentences
        sentences_target: Target sentences
        Q: Rotation matrix
        source_mean: Source centering mean
        target_mean: Target centering mean
        top_k: Show top K best/worst alignments
        
    Returns:
        Dict: Alignment analysis
    """
    from .alignment_utils import apply_alignment
    
    similarities = []
    sentence_list = []
    
    for sentence in sentences.keys():
        if sentence not in sentences_target:
            continue
        
        # Apply alignment
        aligned = apply_alignment(sentences[sentence], Q, source_mean, target_mean)
        
        # Compute similarity
        sim = torch.nn.functional.cosine_similarity(
            aligned.unsqueeze(0),
            sentences_target[sentence].unsqueeze(0)
        ).item()
        
        similarities.append(sim)
        sentence_list.append(sentence)
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)
    
    analysis = {
        "mean_similarity": np.mean(similarities),
        "std_similarity": np.std(similarities),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities),
        "num_alignments": len(similarities),
        "best_alignments": [],
        "worst_alignments": []
    }
    
    # Best alignments
    for idx in sorted_indices[-top_k:]:
        analysis["best_alignments"].append({
            "sentence": sentence_list[idx],
            "similarity": similarities[idx]
        })
    
    # Worst alignments
    for idx in sorted_indices[:top_k]:
        analysis["worst_alignments"].append({
            "sentence": sentence_list[idx],
            "similarity": similarities[idx]
        })
    
    return analysis
