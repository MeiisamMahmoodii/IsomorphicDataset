#!/usr/bin/env python3
"""
Read experiments/results/<run>/metrics.json and print a funnel summary for tables.
Usage:
  python scripts/eval_funnel.py experiments/results/isomorphic_baseline_20260101_120000/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Print funnel metrics from metrics.json")
    p.add_argument("metrics_json", type=Path, help="Path to metrics.json")
    args = p.parse_args()
    data = json.loads(args.metrics_json.read_text())
    funnel = data.get("funnel", {})
    print("=== Funnel ===")
    for k in sorted(funnel.keys()):
        print(f"  {k}: {funnel[k]}")
    print("\n=== Alignment (best pooling) ===")
    print(json.dumps(data.get("alignment", {}), indent=2, default=str))
    print("\n=== Reference gate ===")
    print(json.dumps(data.get("reference_gate", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
    sys.exit(0)
