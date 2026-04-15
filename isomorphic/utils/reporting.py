"""
Experiment reporting: EXPERIMENT_REPORT.md from live metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict


class IsomorphismReporter:
    """Writes markdown summaries from pipeline metrics."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def generate_final_report(self, metrics: Dict[str, Any], datasets: Any) -> Path:
        """Create EXPERIMENT_REPORT.md from metrics and dataset handles."""
        report_path = self.output_dir / "EXPERIMENT_REPORT.md"

        funnel = metrics.get("funnel", {})
        align = metrics.get("alignment", {})
        refg = metrics.get("reference_gate", {})
        rewrite = metrics.get("rewrite_stats", {})

        lines = [
            "# Isomorphism Study: Pipeline Report\n",
            "\n## 1. Funnel\n",
        ]
        for k, v in funnel.items():
            lines.append(f"- **{k}**: {v}\n")

        lines.append("\n## 2. Rewrite / constraint attempts\n")
        lines.append(f"```json\n{json.dumps(rewrite, indent=2)}\n```\n")

        lines.append("\n## 3. Hub alignment (pooling ablation)\n")
        lines.append(f"- **Hub model**: {align.get('hub', 'n/a')}\n")
        lines.append(f"- **Best pooling (mean quality across datasets)**: {align.get('best_pooling_method', 'n/a')}\n")
        scores = align.get("method_scores_avg", {})
        if scores:
            lines.append("\n| Pooling | Mean alignment quality |\n| :--- | :--- |\n")
            for m, s in sorted(scores.items()):
                lines.append(f"| {m} | {s:.6f} |\n")

        per = align.get("per_dataset_method", {})
        if per:
            lines.append("\n### Per-dataset\n")
            lines.append(f"```json\n{json.dumps(per, indent=2, default=str)}\n```\n")

        lines.append("\n## 4. Reference gate (Wasserstein + cosine)\n")
        lines.append(f"```json\n{json.dumps(refg, indent=2)}\n```\n")

        accepted = sum(len(ds._data) for ds in datasets.values()) if isinstance(datasets, dict) else 0
        kept = funnel.get("final_accepted", None)
        lines.append(f"\n## 5. Dataset rows (loaded): {accepted}\n")
        if kept is not None:
            lines.append(f"- **Accepted after gate**: {kept}\n")

        with open(report_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        print(f"[DONE] Research report generated at {report_path}")
        return report_path
