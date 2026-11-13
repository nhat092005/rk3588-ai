# Wave 1 acceptance, 2026-11-13

Experiment id `acceptance-20261113-wave1`. Step 9 of the migration plan: the migrated pipeline reproduces the legacy
baseline `ppe5_yolov8n_baseline_100ep_s0`. Run with the legacy checkout renamed (unreachable), code at ai `d943f05`,
common `5b737d6`, board `rk3588-board` synced from the same commits. Legacy reference files: `legacy-final`, run
`experiments/runs/ppe5_yolov8n_baseline_100ep_s0`. The pipeline itself never read the legacy checkout; the reference side of checks 2a, 2c and 2e (legacy
interpreter and code, legacy calibration PNGs, legacy `best.onnx`) was read from the renamed copy.

| # | Check | Result |
|---|---|---|
| 1 | `make evaluate BACKEND=pt` | `eval/pt_fp32_{css_test,ppe2286_ext}.json` equal legacy in every field except provenance (`repos` replaces `git_commit`, `git_dirty`): map50_95 0.4999 and 0.2624, all per-class APs, full dataset fingerprints. PASS |
| 2a | `make export-npu PREC=int8`: calibration list | 300 images, equal to the list that `legacy-final:ai/core/rknn/calibration.py` gives on the legacy labels. PASS |
| 2b | Calibration list vs legacy `export_int8.json` | Different (40 of 300 common). The legacy file is from 2026-10-30, before legacy commit `13639b3` (2026-11-08) sorted the candidates before the seeded shuffle; until then the list depended on `PYTHONHASHSEED` and cannot be reproduced |
| 2c | ONNX sha256 | Differs only by the `date` metadata of Ultralytics; graph and model without metadata identical to legacy `best.onnx`. PASS |
| 2d | RKNN sha256 | Not comparable: two builds from identical inputs differ in 999 bytes (pointer-like values), RKNN-Toolkit2 2.3.2 output is not byte-deterministic |
| 2e | Board eval of an INT8 RKNN built here with the legacy calibration images | `legacy_calib_rknn_int8_*.json`: map50_95 0.3675 (css_test) and 0.2313 (ppe2286_ext), all per-class APs equal legacy `eval/rknn_int8_*.json`. PASS |
| 2f | Board eval of the official INT8 RKNN (reproducible calibration) | map50_95 0.3543 (css_test, legacy 0.3675) and 0.2342 (ppe2286_ext, legacy 0.2313). The plan criterion (equal to 0.3675 when sha256 differs) FAILS for this RKNN; 2a and 2e show the difference comes from the calibration set only. Substitute criterion pending owner acceptance |
| 3 | `make bench-board PREC=int8` | `bench/tier2_int8_AUTO.json` with the new provenance, frequencies locked: NPU 20.84 ms mean, 43.42 FPS (latency), 7.65 detections per image. PASS (no legacy tier-2 INT8 to compare) |
| 4 | `make golden-update`, `make golden-test` | `experiments/golden/evaluator_v1.json` written, 24 metrics within 0.0005. PASS |
| 5 | `deploy/probes/probe_infer` vs RKNNLite (`probe_check.py`) | 82 css_test images: raw output bit-identical 82/82, `rk3588_common.postprocess` detections identical 82/82. PASS |
| 6 | Leaderboard | `experiments/leaderboards/ppe5.csv`: two rows from check 1, values equal the legacy rows |

Not done: `experiments/index.csv` row (its writer `log_experiment` is not migrated, see `LEGACY.md`). Tier 1 (adb,
`rknn_server`) was not part of the acceptance.
