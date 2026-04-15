# Run guide (single GPU + multi-GPU)

## Prereqs

- **Python**: 3.10+ recommended
- **NVIDIA drivers + CUDA**: required for GPU runs
- **Disk**: models will be downloaded to the Hugging Face cache (can be large)

## Setup

### Create venv + install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### (Optional) choose which GPUs are visible

- **Use GPU 0 and 1**:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

- **Use only GPU 2**:

```bash
export CUDA_VISIBLE_DEVICES=2
```

## Multi-GPU behavior (what changed)

- If a model config has `device_map: "auto"` and you have multiple visible GPUs, the model will be **sharded across GPUs** automatically (Transformers/Accelerate).
- Input tensors are placed on the **correct shard entry device** (so generation/extraction works with sharded models).
- If you have only one visible GPU, `"auto"` behaves like single-GPU placement.

## 3 runnable scripts

### 1) Smoke run (fast sanity check)

Runs `config/smoke_test.yaml` with a **small reference model** + 1 tiny dataset sample.

```bash
python3 scripts/run_smoke.py --output experiments/results
```

If you still see `ModuleNotFoundError: isomorphic`, make sure you’re running from the repo root:

```bash
cd /lts/meisam/IsomorphicDataset
python3 scripts/run_smoke.py --output experiments/results
```

### 2) Rewrite preview (3 small models)

This prints a few rewritten outputs so you can visually inspect results.

```bash
python3 scripts/run_rewrite_preview_3small.py --samples 3 --dataset toxigen --max-words 20
```

### 3) Full production run

Runs `config/default.yaml`.

```bash
python3 scripts/run_production.py --output experiments/results
```

## Running via `main.py` (alternative)

```bash
python3 main.py --config config/smoke_test.yaml --output experiments/results
python3 main.py --config config/default.yaml --output experiments/results
```

## Notes / troubleshooting

- **Out of memory**:
  - set `CUDA_VISIBLE_DEVICES` to use more GPUs (so `device_map: auto` can shard)
  - reduce `max_samples` in the config
  - reduce `max_length`
  - enable `load_in_4bit` by using larger models only when needed
- **First run is slow**: models/datasets download; later runs are faster.
