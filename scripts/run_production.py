import sys
from pathlib import Path as _Path

# Allow running without installing the package (PYTHONPATH not required)
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

from isomorphic.config import ConfigManager
from isomorphic.pipeline import IsomorphicPipeline


def main() -> int:
    p = argparse.ArgumentParser(description="IsomorphicDataSet production run")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--output", type=Path, default=Path("experiments/results"))
    args = p.parse_args()

    cfg = ConfigManager().load_config(args.config)
    cfg.experiment.output_dir = args.output

    results = IsomorphicPipeline(cfg).run()
    print(f"\n[DONE] Results: {results['experiment_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
