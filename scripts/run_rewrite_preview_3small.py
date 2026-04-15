import sys
from pathlib import Path as _Path

# Allow running without installing the package (PYTHONPATH not required)
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

from isomorphic.datasets.base_dataset import DatasetFactory
from isomorphic.datasets.reference_constraints import ReferenceConstraintBuilder
from isomorphic.generation.rewriter import ModelRewriter
from isomorphic.utils.gpu_manager import GPUManager


def main() -> int:
    """
    Quick "what do the rewrites look like?" preview:
    - builds constraints using a small reference model
    - rewrites with 3 small models
    - prints variations for a handful of entries
    """
    p = argparse.ArgumentParser(description="Preview rewrites with 3 small models")
    p.add_argument("--samples", type=int, default=3, help="Number of dataset rows to preview")
    p.add_argument("--dataset", type=str, default="toxigen", help="Dataset type (toxigen/jigsaw/...)")
    p.add_argument("--max-words", type=int, default=20, help="Upper bound for a single length bin (preview)")
    args = p.parse_args()

    # Keep it lightweight by default.
    # Keep reference model widely compatible with current transformers versions.
    reference_model_id = "huihui-ai/Llama-3.2-1B-Instruct-abliterated"
    model_ids = [
        "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
        "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated",
    ]

    gpu = GPUManager()
    device_map = "auto" if gpu.num_gpus > 1 else gpu.get_optimal_device(reference_model_id)

    # Load a tiny dataset subset
    ds = DatasetFactory.create(args.dataset, max_samples=args.samples)
    if not ds.load():
        raise RuntimeError(f"Failed to load dataset: {args.dataset}")
    ds.preprocess()
    ds._data = [e for e in ds._data if isinstance(e.seed_text, str) and e.seed_text.strip()]
    if not ds._data:
        raise RuntimeError("No non-empty seed_text entries available after preprocessing")
    print(f"[INFO] Previewing {len(ds._data)} non-empty entries")

    # Build banned words + rewrite specs (Phase A-like)
    ref = ModelRewriter(reference_model_id, device=str(device_map), device_map=device_map, load_in_4bit=False)
    builder = ReferenceConstraintBuilder(ref.model, ref.tokenizer)
    for entry in ds._data:
        builder.process_entry(entry)
        print(
            f"\n[CONSTRAINTS] seed='{entry.seed_text[:140]}' "
            f"banned_words={entry.forbidden_words}"
        )

    # Rewrite with each model and print
    for mid in model_ids:
        dm = "auto" if gpu.num_gpus > 1 else gpu.get_optimal_device(mid)
        try:
            rw = ModelRewriter(mid, device=str(dm), device_map=dm, load_in_4bit=False)
        except Exception as e:
            print(f"\n[WARN] Skipping model {mid}: {e}")
            continue
        print("\n" + "=" * 90)
        print(f"MODEL: {mid}")
        print("=" * 90)
        for entry in ds._data:
            # Force a single short bin for preview friendliness
            if entry.length_constraints:
                entry.length_constraints = entry.length_constraints[:1]
                entry.length_constraints[0].max_words = args.max_words
            rw.process_entry(entry)
            # Print all variations written for this model
            print(f"\nSEED: {entry.seed_text}")
            for k, v in entry.variations.items():
                if k.startswith(mid):
                    print(f"- {k}: {v}")

        del rw

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
