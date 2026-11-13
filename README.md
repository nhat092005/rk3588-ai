# rk3588-ai

PC side of PPE detection on RK3588: datasets, training, export, evaluation, and the measurement of models on the NPU, including the research harness that runs on the board (`ai/board/`). Rules: meta-repo `STANDARDS.md`.

| Path | Content | Status |
|---|---|---|
| `ai/paths.py` | Every path of this repo (`data/`, `experiments/`, `results/`) | Migrated |
| `ai/core/` | `dataset.py` (dataset.yaml lookup), `trainer.py` (Ultralytics train), `evaluator.py` (Evaluation Protocol v1), `backends.py` (PyTorch/ONNX, RKNN board dump), `provenance.py` (repo states of ai and common) | Migrated |
| `ai/core/rknn/` | `baseline.py` (Ultralytics RKNN and ONNX export), `hybrid.py` (hybrid INT8 recipes, h5 standard), `calibration.py` (calibration sampling, letterboxed PNG through `rk3588_common.model_io.preprocess`), `benchmark.py` (tier-1 `eval_perf`/`eval_memory` from the PC) | Migrated |
| `ai/models/` | One folder per model (`yolov8n_baseline`), see `ai/models/README.md` | Migrated |
| `ai/automation/configs/` | Run configs (`ppe5_yolov8n_baseline`, `_200ep`) | Migrated |
| `ai/automation/pipeline/` | `run` (train + protocol v1 eval + leaderboard), `evaluate` (pt, onnx, rknn on the board), `check_evaluator` (golden, crosscheck), `export_npu`, `complexity` | Migrated |
| `ai/automation/report/`, `ai/automation/bench/` | `tables` (Tables A-M), `bench_gpu`, `benchmark_npu` (tier 1), `board_check` | Migrated |
| `ai/board/` | Research harness that runs on the board: `bench/` (`bench_npu`, `bench_cpu_onnx`, `bench_throughput`, `WARMUP`), `probe/infer_dump` (NPU outputs for the evaluator) | Migrated |
| `ai/research/`, `ai/npu/` | Research lines and NPU studies | Migrated on demand, closed ones stay in `legacy-final` |
| `experiments/`, `results/` | Measurements produced by this repo. `experiments/runs/ppe5_yolov8n_baseline_100ep_s0` is imported from legacy (`imported.json`) | Migrated (baseline only) |
| `data/` | Datasets (only `DATA_CARD.md` tracked): raw `css` and `ppe2286`, `ppe5` rebuilt by `make prepare-data DATASET=ppe5` | Migrated |
| `scripts/` | `logtee.py` (command log with repo states), `data/prepare_ppe5.py` (ppe5 from css + ppe2286, relative links, no `path:` in dataset.yaml) | Migrated |
| `tests/` | Equivalence tests against legacy fixtures | |
| `Makefile` | Targets of this repo (`make help`), board settings from the meta-repo `board.mk` | Migrated |
| `LEGACY.md` | Map of old experiment codes, code, data and runs that stayed in `legacy-final` | |

Model I/O constants (`IMGSZ`, `PROTOCOL_VERSION`), pre/post-processing, telemetry and the provenance schema live in `rk3588-common`. The evaluator's own NMS settings are `EVAL_CONF`, `EVAL_IOU`, `EVAL_MAX_DET` (Ultralytics val settings), a separate contract from the deployment NMS in `rk3588_common.model_io`.

## Running

Work from the `rk3588` meta-repo checkout (it pins `rk3588-common` and holds the `.venv`, `make sync`). Run commands from this directory: `../.venv/bin/python -m ai.<module>`. The project is not installed, so from the meta-repo root `ai` would resolve to this directory instead of the package.

## Board

`make sync-board` copies `ai/board`, `common` and the deploy scripts (meta-repo `make sync-board-py`) and the test images to `~/<BOARD_DIR>/ai` on the board, where board scripts run as `../.venv/bin/python -m ai.board...`. The board has no git checkout: result files get the PC repo states through `--repo NAME:COMMIT:DIRTY` (`python -m ai.core.provenance board-args`).

## Tests

`../.venv/bin/python -m pytest tests`. `tests/test_core.py` and `tests/test_rknn.py` check `ai.core` against fixtures written once by `tests/make_fixtures.py` with the legacy code at tag `legacy-final` (see its docstring). Real inputs from legacy: an `eval_perf` debug output and the step-1 quantization cfg of the SiLU baseline. `tests/test_models.py` checks every config against its model folder.
