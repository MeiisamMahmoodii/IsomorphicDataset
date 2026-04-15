"""
Preflight: load each configured rewrite model once, run one constrained rewrite
on a fixed neutral sentence, report load + generation status.

Usage:
  python scripts/check_models_rewrite.py
  python scripts/check_models_rewrite.py --config config/full_fleet_trevor_reference.yaml
  python scripts/check_models_rewrite.py --models "a/b,c/d"
"""

import gc
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from typing import List, Tuple

import torch

from isomorphic.config import ConfigManager
from isomorphic.datasets.base_dataset import LengthConstraint
from isomorphic.generation.constraint_utils import rewrite_passes_constraints
from isomorphic.generation.rewriter import ModelRewriter
from isomorphic.utils.gpu_manager import GPUManager


# Neutral test sentence; banned words chosen so a valid rewrite is usually possible.
DEFAULT_SENTENCE = "The weather is pleasant today in the coastal region."
# Must match pipeline-style bans (3–5 content words) + global list in ModelRewriter
DEFAULT_BANNED = ["weather", "pleasant", "coastal"]


def _load_model_ids_from_config(path: _Path) -> List[str]:
    cfg = ConfigManager().load_config(path)
    return [m.model_id for m in cfg.experiment.models]


def _needs_4bit(model_id: str) -> bool:
    return any(tag in model_id for tag in ("31B", "35B", "26B", "20b", "20B", "27B"))


def main() -> int:
    p = argparse.ArgumentParser(description="Preflight: one rewrite per model")
    p.add_argument(
        "--config",
        type=_Path,
        default=_Path("config/full_fleet_trevor_reference.yaml"),
        help="YAML with experiment.models (model_id list).",
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Override: comma-separated Hugging Face model ids (skips --config).",
    )
    p.add_argument("--sentence", type=str, default=DEFAULT_SENTENCE)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=20)
    args = p.parse_args()

    if args.models.strip():
        model_ids = [x.strip() for x in args.models.split(",") if x.strip()]
    elif args.config.exists():
        model_ids = _load_model_ids_from_config(args.config)
    else:
        model_ids = list(ModelRewriter.MODELS)

    if not model_ids:
        print("No models to test.")
        return 1

    constraint = LengthConstraint(min_words=args.min_words, max_words=args.max_words)
    # Effective bans = per-sentence + ModelRewriter global generics (same as pipeline)
    effective_banned: List[str] = []
    seen = set()
    for w in list(DEFAULT_BANNED) + list(ModelRewriter.GLOBAL_GENERIC_BANNED_WORDS):
        n = str(w).strip().lower()
        if n and n not in seen:
            effective_banned.append(n)
            seen.add(n)

    gpu = GPUManager()
    print("=" * 80)
    print("MODEL REWRITE PREFLIGHT")
    print("=" * 80)
    print(f"Sentence: {args.sentence}")
    print(f"Banned (effective): {effective_banned}")
    print(f"Length: {args.min_words}-{args.max_words} words")
    print(f"Models to test: {len(model_ids)}")
    print()

    results: List[Tuple[str, str, str, str]] = []

    for i, mid in enumerate(model_ids, 1):
        dm = "auto" if gpu.num_gpus > 1 else gpu.get_optimal_device(mid)
        load_4bit = _needs_4bit(mid)
        status_load = "ok"
        status_gen = "—"
        detail = ""
        out = ""

        try:
            rw = ModelRewriter(
                mid,
                device=str(dm),
                device_map=dm,
                load_in_4bit=load_4bit,
                verbose_attempts=False,
            )
        except Exception as e:
            status_load = "FAIL"
            detail = str(e)[:200]
            results.append((mid, status_load, status_gen, detail))
            print(f"[{i}/{len(model_ids)}] {mid}")
            print(f"  LOAD: {status_load} — {detail}")
            print()
            continue

        try:
            prompt = ModelRewriter.build_rewrite_prompt(
                args.sentence, effective_banned, constraint
            )
            out = rw.rewrite_with_prompt(prompt, constraint)
            ok, reason = rewrite_passes_constraints(
                out, effective_banned, args.min_words, args.max_words
            )
            if ok:
                status_gen = "OK"
                detail = "constraints_pass"
            elif out and str(out).strip():
                status_gen = "OUT"
                detail = reason
            else:
                status_gen = "EMPTY"
                detail = reason or "empty"
        except Exception as e:
            status_gen = "FAIL"
            detail = str(e)[:200]
        finally:
            del rw
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append((mid, status_load, status_gen, detail))
        print(f"[{i}/{len(model_ids)}] {mid}")
        print(f"  LOAD: {status_load}  GEN: {status_gen}  ({detail})")
        if out and str(out).strip():
            preview = str(out).strip().replace("\n", " ")[:220]
            print(f"  Output: {preview}")
        print()

    ok_load = sum(1 for _, L, _, _ in results if L == "ok")
    ok_gen = sum(1 for _, _, G, _ in results if G == "OK")
    print("=" * 80)
    print(f"Summary: loaded {ok_load}/{len(results)}  |  constraint-OK rewrites {ok_gen}/{len(results)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
