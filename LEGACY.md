# Legacy map

The old repo is frozen at tag `legacy-final` (commit `1bc04d3`, GitHub `nhat092005/rk3588-legacy`). This file maps
what stayed there. Read a legacy file with `git -C <legacy checkout> show legacy-final:<path>`. Old experiment codes
appear only in this file (meta-repo `STANDARDS.md`, section 3).

Lines of work below migrate right before work on them resumes, each as its own step. Closed experiments stay in
legacy, code and data.

## Experiment codes

Legacy ids are the `exp_id` column of `experiments/index.csv`. Results of an experiment are in its directory, its
analysis scripts in the `script` column (mostly `ai/npu/<topic>/`).

| Codes | Domain / topic | Legacy directory (under `experiments/`) | Legacy code |
|---|---|---|---|
| `onboard`, `e2`, `e2b`, `e2c`, `e2d`, `e2e` | npu / latency | `npu/latency/onboard`, `npu/latency/onboard_e2*` | `ai/npu/latency/analyze_onboard.py` |
| `x1`-`x6`, `x8`, `x9`, `x10` | npu / latency | `npu/latency/onboard_x*`, `npu/latency/analysis_x10` | `ai/npu/latency/analyze_*.py`, `aggregate_layers.py` |
| `submit_trace` | npu / latency | `npu/latency/submit_trace` | in the directory (`npu_trace.c`, `summarize.py`) |
| `g4`, `g4b`, `g4v`, `g4v2` (gate G4) | npu / operator_support | `npu/operator_support/eval_g4`, `onboard_g4*` | `ai/npu/operator_support/analyze_g4*.py`, profile `ai/automation/profiles/npu_g4.yaml`, `ai/automation/report/acceptance.py` |
| `x7`, `x7b` | npu / operator_support | `npu/operator_support/eval_x7*` | `ai/npu/operator_support/analyze_x7.py` |
| `q2c`, `q2e`, `q2e_qat`, `q2h`, `q2i`, `q2j` | npu / operator_support | `npu/operator_support/onboard_q2*` | `ai/npu/operator_support/analyze_q2*.py`, `build_*.py`, `ai/core/rknn/{split_head,qat_split,blocks}.py` |
| `c5_build`, `c3_vs_c2f` | npu / operator_support | `npu/operator_support/c5_build`, `c3_vs_c2f` | in the directory, `ai/npu/operator_support/analyze_c3_vs_c2f.py` |
| `thr`, `q2f_thr`, `q2f_split`, `input_384x640` | npu / throughput | `npu/throughput/onboard_thr`, `onboard_q2f_*`, `input_384x640` | `ai/npu/throughput/analyze_*.py`, `ai/board/bench/bench_throughput_rep.py` |
| `a1`-`a7`, `a9`, `a9b`, `a9c`, `a9d` | research / all_core | `research/all_core/a*` | `ai/research/all_core/`, models `ai/models/allcore_*` |
| camera `2026-11-07`, `2026-11-08` runs (`camera_runs`, `ab_runs`, `cam_fps`, `enc_rga`, `distort`, `stretch`, `isp_record`, `rga_caps`, `pipeline_2026-11-07`) | camera | `camera/*` | `ai/board/app/camera.py`, scripts in each directory |

## Code not migrated

| Legacy path | Content |
|---|---|
| `ai/npu/`, `ai/research/all_core/`, `ai/research/modular/`, `ai/research/notebooks/` | NPU studies and research lines above |
| `ai/models/` (all but `yolov8n_baseline`), their configs in `ai/automation/configs/` | ReLU, LeakyReLU, ReLU6, no-DFL, C3, scratch, all-core, yolov8s and yolo26n models |
| `ai/automation/profiles/`, `ai/automation/report/acceptance.py` | Gate G4 acceptance (conditions A-C) |
| `ai/automation/report/compare_train_sets.py` | Comparison of training sets (ppe5, ppe4, ppe2286, ppe5_plus) on common test sets |
| `ai/core/rknn/{deploy,split_head,qat_split,blocks}.py`, `ai/board/probe/{probe_runtime,split_gate}.py`, `ai/board/bench/{bench_decode,bench_throughput_rep}.py`, `ai/board/bench/rknn_perf_bench.c` | Split head, QAT, per-block and decode studies |
| `ai/board/app/`, `scripts/demo/` | Camera app and demos (product app: `rk3588-deploy` by the roles of `STANDARDS.md`) |
| `ai/core/provenance.py:log_experiment` | Row writer of `experiments/index.csv` (no wave 1 caller) |
| `scripts/data/prepare_{ppe2286,ppe4,ppe5_plus}.py`, `scripts/data/common.py` | Datasets `ppe2286` (processed), `ppe4`, `ppe5_plus` |
| `scripts/archive/`, `scripts/npu/` | Archive helpers, RKNN build log analysis |

Moved elsewhere: `ai/board/common.py` and `ai/board/split_decode.py` to `rk3588_common.model_io` and `board_env`,
`ai/board/sh/` and `ai/board/requirements.txt` to `rk3588-deploy`.

## Data and runs

| Legacy | Here |
|---|---|
| `data/css/raw`, `data/ppe2286/raw`, data cards of css, ppe2286, ppe5 | Copied unchanged (`ppe2286` card notes what was not migrated) |
| `data/ppe5/processed` | Rebuilt by `make prepare-data DATASET=ppe5`: labels byte-identical, fingerprints of `css_test` and `ppe2286_ext` equal to legacy, relative links, no `path:` |
| `data/ppe2286/processed`, `data/ppe4`, `data/ppe5_plus`, `data/demo` | Not migrated |
| `experiments/runs/ppe5_yolov8n_baseline_100ep_s0` | `best.pt`, `config.yaml`, `args.yaml` (paths made relative), `metrics.csv`, with `imported.json` (sha256 of the legacy files) |
| the other 33 runs, `experiments/leaderboards/`, `experiments/index.csv`, `experiments/golden/` | Not migrated; leaderboard and index start empty |
