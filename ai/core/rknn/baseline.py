"""Baseline RKNN export using Ultralytics native RKNN backend."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics import YOLO


@dataclass
class ExportConfig:
    weights: str
    platform: str = "rk3588"
    imgsz: int = 640
    quantize: int = 8  # 8 for INT8 (w8a8), 16 for FP16
    calib_images_dir: str | None = None  # letterboxed calibration images, required for INT8
    # hybrid recipe of ai.core.rknn.hybrid (h5 is the project standard INT8), INT8 only; None = Ultralytics default
    recipe: str | None = None


def export_onnx(weights: str, imgsz: int = 640) -> Path:
    """Plain FP32 ONNX (pixel box coordinates), same opset as the RKNN export. Used on the board CPU."""
    return Path(YOLO(weights).export(format="onnx", imgsz=imgsz, opset=19))


def export_rknn(cfg: ExportConfig, out_dir: Path) -> tuple[Path, list[str]]:
    """Export to RKNN and move Ultralytics' <weights>_rknn_model/ to out_dir.

    Ultralytics writes every precision to the same <weights>_rknn_model/ folder and,
    for INT8, overwrites then deletes <weights>.onnx (engine/exporter.py export_rknn),
    so call export_onnx() after this function, not before.
    """
    model = YOLO(cfg.weights)
    kwargs = {}
    float16_layers: list[str] = []
    import ultralytics.utils.export.rknn as urknn

    original = urknn.onnx2rknn
    if cfg.recipe:
        if cfg.quantize != 8:
            raise ValueError("a hybrid recipe needs quantize=8")
        float16_layers = _patch_onnx2rknn(cfg.recipe)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if cfg.quantize == 8:
                if not cfg.calib_images_dir:
                    raise ValueError("INT8 export needs calib_images_dir")
                # Ultralytics needs calib images via a dataset split: build a throwaway dataset.yaml
                # (train/val keys required but unused).
                calib = str(Path(cfg.calib_images_dir).resolve())
                data_yaml = Path(tmp) / "calib.yaml"
                data_yaml.write_text(
                    yaml.safe_dump({"path": tmp, "train": calib, "val": calib, "calib": calib, "names": model.names})
                )
                kwargs = {"data": str(data_yaml), "split": "calib"}
            result = Path(
                model.export(format="rknn", name=cfg.platform, quantize=cfg.quantize, imgsz=cfg.imgsz, **kwargs)
            )
    finally:
        urknn.onnx2rknn = original  # a hybrid export must not leak into a later default export in the same process

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.move(str(result), str(out_dir))
    return out_dir, float16_layers


def _patch_onnx2rknn(recipe: str) -> list[str]:
    """Replace Ultralytics' onnx2rknn by a hybrid build of the same ONNX (normalized coordinates, same config and
    calibration list).

    The returned list is filled with the float16 tensor names when the export runs. Ultralytics' own code writes the
    folder, dataset.txt is read from it, and metadata.yaml is saved exactly as in onnx2rknn.
    """
    import ultralytics.utils.export.rknn as urknn
    from ultralytics.utils import YAML

    from ai.core.rknn.hybrid import build_hybrid

    layers: list[str] = []

    def onnx2rknn(onnx_file, output_dir, name="rk3588", quantize=8, batch=1, dataset=None, metadata=None, prefix=""):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        layers[:] = build_hybrid(
            onnx_file, dataset, output_dir / f"{Path(onnx_file).stem}-{name}.rknn", recipe, batch=batch, target=name
        )
        if metadata:
            YAML.save(output_dir / "metadata.yaml", metadata)
        return str(output_dir)

    urknn.onnx2rknn = onnx2rknn
    return layers
