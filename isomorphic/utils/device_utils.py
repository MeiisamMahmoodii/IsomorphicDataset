"""
Device helpers for single-GPU, multi-GPU sharding, and CPU.

Transformers with `device_map="auto"` shards modules across multiple GPUs.
In that case, you must place inputs on the device of the *first* shard
instead of calling `.to("auto")` or relying on `model.device`.
"""

from __future__ import annotations

from typing import Any

import torch


def get_model_input_device(model: Any) -> torch.device:
    """
    Best-effort: find a device that accepts input tensors for `model`.

    - For standard (single device) models, returns `model.device` or first param device.
    - For sharded models (Accelerate dispatch), returns the first parameter's device.
    - Falls back to CPU if nothing is available.
    """
    dev = getattr(model, "device", None)
    if dev is not None:
        try:
            return torch.device(dev)
        except Exception:
            pass

    try:
        p = next(model.parameters())
        return p.device
    except Exception:
        return torch.device("cpu")


def normalize_device_map(device_or_map: Any) -> Any:
    """
    Normalize a user/config-provided `device_map` argument.

    Accepts:
    - "auto" (multi-GPU sharding)
    - "cpu"
    - "cuda:0" / "cuda:1" ... (maps whole model to one GPU)
    - dict-style accelerate device maps (returned unchanged)
    """
    if isinstance(device_or_map, dict):
        return device_or_map

    if device_or_map is None:
        return None

    if not isinstance(device_or_map, str):
        return device_or_map

    s = device_or_map.strip().lower()
    if s in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return s
    if s == "cpu":
        return {"": "cpu"}
    if s.startswith("cuda"):
        # map the whole model to a single GPU index
        # accelerate expects an int GPU id for CUDA devices
        try:
            idx = int(s.split(":")[1]) if ":" in s else 0
        except Exception:
            idx = 0
        return {"": idx}

    return device_or_map
