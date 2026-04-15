# Isomorphism Study: Pipeline Report

## 1. Funnel
- **after_phase_a_entries**: 2
- **models_vectorized**: ['Qwen2.5-0.5B']
- **final_accepted**: 0

## 2. Rewrite / constraint attempts
```json
{
  "Qwen2.5-0.5B": {
    "attempts_total": 19,
    "constraint_passes_recorded": 1,
    "max_attempts_per_slot": 5
  }
}
```

## 3. Hub alignment (pooling ablation)
- **Hub model**: n/a
- **Best pooling (mean quality across datasets)**: n/a

## 4. Reference gate (Wasserstein + cosine)
```json
{
  "pairs_judged": 4,
  "entries_failed_gate": 2,
  "wasserstein_max": 0.98,
  "cosine_min": 0.75
}
```

## 5. Dataset rows (loaded): 2
- **Accepted after gate**: 0
