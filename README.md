# rk3588-ai

PC side of PPE detection on RK3588: datasets, training, export, evaluation, and the measurement of models on the NPU, including the research harness that runs on the board (`ai/board/`). Rules: meta-repo `STANDARDS.md`.

| Path | Content | Status |
|---|---|---|
| `ai/paths.py` | Every path of this repo (`data/`, `experiments/`, `results/`) | Migrated |
| `ai/core/` | `dataset.py` (dataset.yaml lookup), `trainer.py` (Ultralytics train), `evaluator.py` (Evaluation Protocol v1), `backends.py` (PyTorch/ONNX, RKNN board dump), `provenance.py` (repo states of ai and common) | Migrated |
| `ai/core/rknn/` | RKNN export, calibration, tier-1 benchmark | Planned (step 6) |
| `ai/models/` | One folder per model | Planned (step 6) |
| `ai/automation/` | Training and evaluation pipeline, configs, reports, PC benches | Planned (step 7) |
| `ai/board/` | Research harness that runs on the board (tier-2 bench, NPU dump) | Planned (step 7) |
| `ai/research/`, `ai/npu/` | Research lines and NPU studies | Migrated on demand, closed ones stay in `legacy-final` |
| `experiments/`, `results/` | Measurements produced by this repo | Planned (step 8) |
| `data/` | Datasets (only `DATA_CARD.md` tracked) | Planned (step 8) |
| `scripts/`, `tests/` | Helper scripts and tests | `tests/` started |

Model I/O constants (`IMGSZ`, `PROTOCOL_VERSION`), pre/post-processing, telemetry and the provenance schema live in `rk3588-common`. The evaluator's own NMS settings are `EVAL_CONF`, `EVAL_IOU`, `EVAL_MAX_DET` (Ultralytics val settings), a separate contract from the deployment NMS in `rk3588_common.model_io`.

## Running

Work from the `rk3588` meta-repo checkout (it pins `rk3588-common` and holds the `.venv`, `make sync`). Run commands from this directory: `../.venv/bin/python -m ai.<module>`. The project is not installed, so from the meta-repo root `ai` would resolve to this directory instead of the package.

## Tests

`../.venv/bin/python -m pytest tests`. `tests/test_core.py` checks `ai.core` against fixtures written once by `tests/make_fixtures.py` with the legacy code at tag `legacy-final` (see its docstring).
