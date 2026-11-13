"""CLI to export a trained run to RKNN FP16 or INT8, plus the plain FP32 ONNX.

Writes runs/<run_id>/weights/rknn_<precision>/ (.rknn + metadata, git-ignored) and best.onnx
(board CPU baseline, Table H), plus tracked runs/<run_id>/export_<precision>.json.
"""

import argparse
import hashlib
import json
import os
import shutil
from importlib.metadata import version
from pathlib import Path

import yaml

from ai.core.dataset import load_dataset_yaml
from ai.core.provenance import code_repos
from ai.core.rknn.baseline import ExportConfig, export_onnx, export_rknn
from ai.core.rknn.calibration import sample_calibration_images, write_letterboxed_calib
from ai.paths import RUNS_DIR
from rk3588_common.provenance import file_sha256, provenance

# Calibration set (protocol v1): 300 train images, >=20/class, seed 0.
# 300 avoids Ultralytics' ">300 images recommended" INT8 warning (exporter.py get_int8_calibration_dataloader).
CALIB_N = 300
CALIB_MIN_PER_CLASS = 20
CALIB_SEED = 0


def build_calibration(dataset: str, out_dir: Path) -> dict:
    data = load_dataset_yaml(dataset)
    images_dir = Path(data["path"]) / data["train"]
    labels_dir = Path(str(images_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
    stems = sorted(sample_calibration_images(labels_dir, n=CALIB_N, min_per_class=CALIB_MIN_PER_CLASS, seed=CALIB_SEED))
    by_stem = {p.stem: p for p in images_dir.iterdir()}
    shutil.rmtree(out_dir, ignore_errors=True)  # no leftovers from a previous export
    write_letterboxed_calib([by_stem[s] for s in stems], out_dir)
    return {
        "source": f"{dataset}/train",
        "n": len(stems),
        "min_per_class": CALIB_MIN_PER_CLASS,
        "seed": CALIB_SEED,
        "preprocess": "LetterBox 640 (same as evaluator), PNG",
        "images": stems,
    }


def reuse_calibration(src_run: str, out_dir: Path) -> dict:
    """Copy another run's calibration images, so every compared model quantizes with the same list (standard G3)."""
    src = json.loads((RUNS_DIR / src_run / "export_int8.json").read_text())["calibration"]
    src_dir = RUNS_DIR / src_run / "weights" / "calib_int8"
    if (
        src_dir.resolve() != out_dir.resolve()
    ):  # exporting the source run itself must not delete its own calibration images
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.copytree(src_dir, out_dir)
    pngs = sorted((out_dir / "images").glob("*.png"))
    assert [p.stem for p in pngs] == sorted(src["images"]), "copied calibration images differ from the source list"
    return {
        **src,
        "reused_from": src_run,
        "stems_md5": hashlib.md5("\n".join(sorted(src["images"])).encode()).hexdigest(),
        "png_md5": hashlib.md5("\n".join(f"{p.name} {file_sha256(p)}" for p in pngs).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="an existing directory under experiments/runs/")
    parser.add_argument("--quantize", type=int, choices=[8, 16], default=8, help="8 = INT8 (w8a8), 16 = FP16")
    parser.add_argument(
        "--calib-from", help="INT8 only: run id whose calibration images are reused instead of sampling"
    )
    parser.add_argument(
        "--recipe",
        choices=["h5"],
        help="INT8 only: hybrid recipe, written as precision int8h5 (float16 box-decode tail, the standard INT8 of phase 2c)",
    )
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    weights_dir = run_dir / "weights"
    train_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if args.recipe and args.quantize != 8:
        parser.error("--recipe needs --quantize 8")
    precision = ("int8h5" if args.recipe else "int8") if args.quantize == 8 else "fp16"
    out_dir = weights_dir / f"rknn_{precision}"

    calib = None
    calib_dir = None
    if args.quantize == 8:
        calib_dir = weights_dir / "calib_int8"  # shared by int8 and int8h5, the same calibration images
        calib = (
            reuse_calibration(args.calib_from, calib_dir)
            if args.calib_from
            else build_calibration(train_cfg["dataset"], calib_dir)
        )

    _, float16_layers = export_rknn(
        ExportConfig(
            weights=str(weights_dir / "best.pt"),
            quantize=args.quantize,
            calib_images_dir=str(calib_dir / "images") if calib_dir else None,
            recipe=args.recipe,
        ),
        out_dir,
    )
    onnx_path = export_onnx(str(weights_dir / "best.pt"))
    rknn_path = next(out_dir.glob("*.rknn"))

    record = {
        "run_id": args.run_id,
        "precision": precision,
        "export_tier": "baseline_native_api",
        "rknn_path": str(rknn_path.relative_to(run_dir)),
        "rknn_size_mb": round(rknn_path.stat().st_size / 2**20, 3),
        "rknn_sha256": file_sha256(rknn_path),
        "onnx_path": str(onnx_path.relative_to(run_dir)),
        "onnx_size_mb": round(onnx_path.stat().st_size / 2**20, 3),
        "onnx_sha256": file_sha256(onnx_path),
        **provenance(code_repos()),
        "ultralytics_version": version("ultralytics"),
        "rknn_toolkit2_version": version("rknn-toolkit2"),
        "onnx_version": version("onnx"),
        "onnxslim_version": version("onnxslim"),
        "calibration": calib,
        "recipe": args.recipe,
        "float16_layers": float16_layers,
    }
    out_path = run_dir / f"export_{precision}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
