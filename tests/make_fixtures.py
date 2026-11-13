"""Write the expected outputs of ai.core (incl. ai.core.rknn) from the legacy code at tag legacy-final.

Run once with the legacy interpreter from this repo's root, never imported by the tests:
    <legacy>/.venv/bin/python tests/make_fixtures.py <legacy>
<legacy> is the legacy checkout (HEAD at legacy-final, ai/core clean).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cases
import numpy as np


def check_legacy(legacy: Path) -> None:
    tag = subprocess.check_output(["git", "-C", str(legacy), "describe", "--exact-match", "--tags", "HEAD"], text=True)
    assert tag.strip() == "legacy-final", f"legacy HEAD is at {tag.strip()}, not legacy-final"
    dirty = subprocess.check_output(["git", "-C", str(legacy), "status", "--porcelain", "--", "ai/core"], text=True)
    assert not dirty, f"legacy ai/core has local changes:\n{dirty}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    legacy = parser.parse_args().legacy.resolve()
    check_legacy(legacy)
    sys.path.insert(0, str(legacy))
    from ai.core import backends as old_backends  # legacy-final:ai/core/backends.py
    from ai.core import evaluator as old  # legacy-final:ai/core/evaluator.py

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        ims, lbs = cases.write_dataset_files(Path(tmp))
        out["fingerprint"] = np.array(old.dataset_fingerprint(ims, lbs))
    for case in range(4):
        out[f"match{case}"] = old.match_predictions(*cases.match_inputs(case))
    for quantize, split in ((8, False), (16, False), (8, True)):
        with tempfile.TemporaryDirectory() as tmp:
            img = cases.write_dump(Path(tmp), quantize, split)
            backend = old_backends.RknnDumpBackend(Path(tmp) / "dump.npz", tmp, old.CONF)
            out[f"dump_q{quantize}_s{int(split)}"] = backend(img, im_file="images/x.jpg").numpy()
    out["eval_sets"] = np.array([f"{k}={old.EVAL_SETS[k]}" for k in ("css_test", "ppe2286_ext")])
    out["eval_sets_ppe5"] = np.array(old.EVAL_SETS_BY_DATASET["ppe5"])
    out["constants"] = np.array(
        [old.PROTOCOL_VERSION, old.IMGSZ, old.CONF, old.IOU, old.MAX_DET, old.MAX_TIME_IMG], dtype=str
    )
    out["iou_thresholds"] = old.IOU_THRESHOLDS.numpy()
    cases.FIXTURES.mkdir(exist_ok=True)
    np.savez_compressed(cases.CORE_EXPECTED, **out)
    print(f"wrote {cases.CORE_EXPECTED.name} ({len(out)} arrays)")
    write_rknn(legacy)
    write_pipeline(legacy)


def write_pipeline(legacy: Path) -> None:
    """make steps of phase2 and phase4 for ppe5 (phase3 depends on the leaderboard content)."""
    import os

    from ai.automation.pipeline import pipeline as old  # legacy-final:ai/automation/pipeline/pipeline.py

    cwd = os.getcwd()
    os.chdir(legacy)  # run_id() runs run.py --print-run-id from the repo root
    try:
        steps = {
            p: [[cmd, str(done.relative_to(old.REPO_ROOT)) if done else None] for cmd, done in old.PHASES[p]("ppe5")]
            for p in ("phase2", "phase4")
        }
    finally:
        os.chdir(cwd)
    cases.PIPELINE_EXPECTED.write_text(json.dumps(steps, indent=1) + "\n")
    print(f"wrote {cases.PIPELINE_EXPECTED.name}")


def git_show(legacy: Path, path: str) -> str:
    return subprocess.check_output(["git", "-C", str(legacy), "show", f"legacy-final:{path}"], text=True)


def write_rknn(legacy: Path) -> None:
    import yaml

    from ai.core.rknn import benchmark as old_bench  # legacy-final:ai/core/rknn/benchmark.py
    from ai.core.rknn import calibration as old_calib  # legacy-final:ai/core/rknn/calibration.py
    from ai.core.rknn import hybrid as old_hybrid  # legacy-final:ai/core/rknn/hybrid.py

    cases.EVAL_PERF_TEXT.write_text(git_show(legacy, "tests/fixtures/eval_perf_debug_int8_core0.txt"))
    cfg = yaml.safe_load(git_show(legacy, "experiments/npu/operator_support/onboard_q2c/step1_silu.quantization.cfg"))
    qp = {k: {"dtype": v.get("dtype")} for k, v in cfg["quantize_parameters"].items()}
    cases.STEP1_DTYPES.write_text(json.dumps(qp, indent=0, sort_keys=True) + "\n")

    text = cases.EVAL_PERF_TEXT.read_text()
    m = old_bench.LAYERS_TIME_RE.search(text)
    expected = {
        "layers": old_bench.parse_layers(text),
        "frequencies": old_bench.parse_frequencies(text),
        "layers_total_us": float(m.group(1)) if m else None,
        "core_mask": sorted(old_bench.CORE_MASK),
        "perf_repeats": old_bench.PERF_REPEATS,
        "recipes": list(old_hybrid.RECIPES),
        "select_layers": {},
        # float16 tensors of the real h5 export of the baseline run
        "h5_export_record": json.loads(
            git_show(legacy, "experiments/runs/ppe5_yolov8n_baseline_100ep_s0/export_int8h5.json")
        )["float16_layers"],
    }
    for recipe in old_hybrid.RECIPES:
        try:
            expected["select_layers"][recipe] = old_hybrid.select_layers(recipe, qp)
        except AssertionError as e:
            expected["select_layers"][recipe] = f"AssertionError: {e}"
    with tempfile.TemporaryDirectory() as tmp:
        labels = cases.write_calib_labels(Path(tmp) / "labels")
        expected["calib_sample"] = old_calib.sample_calibration_images(labels, n=30, min_per_class=4, seed=0)
        expected["calib_sample_default"] = old_calib.sample_calibration_images(labels)
        srcs = cases.write_calib_images(Path(tmp) / "src")
        images = old_calib.write_letterboxed_calib(srcs, Path(tmp) / "out")
        import cv2

        expected["calib_png"] = {
            p.stem: cases.digest(cv2.imread(str(images / f"{p.stem}.png"), cv2.IMREAD_UNCHANGED)) for p in srcs
        }
    cases.RKNN_EXPECTED.write_text(json.dumps(expected, indent=1) + "\n")
    print(f"wrote {cases.RKNN_EXPECTED.name}, {cases.EVAL_PERF_TEXT.name}, {cases.STEP1_DTYPES.name}")


if __name__ == "__main__":
    main()
