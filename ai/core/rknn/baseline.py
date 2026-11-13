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
    """Configuration for baseline or hybrid RKNN export."""

    weights: str
    platform: str = "rk3588"
    imgsz: int = 640
    quantize: int = 8  # 8 for INT8 (w8a8), 16 for FP16
    calib_images_dir: str | None = (
        None  # letterboxed calibration images, required for INT8
    )
    # hybrid recipe of ai.core.rknn.hybrid (h5 is standard INT8),
    # INT8 only; None = Ultralytics default
    recipe: str | None = None


def export_onnx(weights: str, imgsz: int = 640) -> Path:
    """Export plain FP32 ONNX (pixel boxes, opset 19) for board CPU."""
    return Path(YOLO(weights).export(format="onnx", imgsz=imgsz, opset=19))


def export_rknn(cfg: ExportConfig, out_dir: Path) -> tuple[Path, list[str]]:
    """Export to RKNN and move <weights>_rknn_model/ to out_dir.

    Ultralytics writes every precision to <weights>_rknn_model/ and,
    for INT8, overwrites then deletes <weights>.onnx, so call
    export_onnx() after this function, not before.
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
                # Ultralytics needs calib images via a dataset split:
                # build a throwaway dataset.yaml (train/val required).
                calib = str(Path(cfg.calib_images_dir).resolve())
                data_yaml = Path(tmp) / "calib.yaml"
                data_yaml.write_text(
                    yaml.safe_dump(
                        {
                            "path": tmp,
                            "train": calib,
                            "val": calib,
                            "calib": calib,
                            "names": model.names,
                        }
                    )
                )
                kwargs = {"data": str(data_yaml), "split": "calib"}
            result = Path(
                model.export(
                    format="rknn",
                    name=cfg.platform,
                    quantize=cfg.quantize,
                    imgsz=cfg.imgsz,
                    **kwargs,
                )
            )
    finally:
        # hybrid export must not leak into a later default export
        urknn.onnx2rknn = original

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.move(str(result), str(out_dir))
    return out_dir, float16_layers


def _patch_onnx2rknn(recipe: str) -> list[str]:
    """Patch Ultralytics onnx2rknn with a hybrid build of the ONNX.

    The returned list is filled with float16 tensor names when the
    export runs. Ultralytics writes the folder, dataset.txt is read
    from it, and metadata.yaml is saved as in onnx2rknn.
    """
    import ultralytics.utils.export.rknn as urknn
    from ultralytics.utils import YAML

    from ai.core.rknn.hybrid import build_hybrid

    layers: list[str] = []

    def onnx2rknn(
        onnx_file: str | Path,
        output_dir: str | Path,
        name: str = "rk3588",
        quantize: int = 8,
        batch: int = 1,
        dataset: str | Path | None = None,
        metadata: dict | None = None,
        prefix: str = "",
    ) -> str:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        layers[:] = build_hybrid(
            onnx_file,
            dataset,
            output_dir / f"{Path(onnx_file).stem}-{name}.rknn",
            recipe,
            batch=batch,
            target=name,
        )
        if metadata:
            YAML.save(output_dir / "metadata.yaml", metadata)
        return str(output_dir)

    urknn.onnx2rknn = onnx2rknn
    return layers
