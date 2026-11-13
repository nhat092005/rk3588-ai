"""Hybrid INT8 builds: named tensors stay float16, the rest INT8.

Recipe h5 is the project standard; the others were used to diagnose
the INT8 accuracy loss. Tensor names match the step-1 quantization cfg
of RKNN-Toolkit2 2.3.2 for the Ultralytics YOLOv8n detect graph with
normalized box coordinates.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import yaml

H1 = re.compile(r"^/model/model\.22/cv3\.\d/cv3\.\d\.2/Conv_output_0$")
H2 = re.compile(r"^/model/model\.22/cv3\.")
H6B = re.compile(r"^/model/model\.22/cv2\.")
BOX_TAIL = re.compile(
    r"^(/model/model\.22/(dfl/.*|Slice_\d*_?output_0-rs"
    r"|Slice_output_0-rs|Sub_output_0-rs|Add_1_output_0-rs"
    r"|Sub_1_output_0-rs|Add_2_output_0-rs|Div_1_output_0-rs"
    r"|Concat_2_output_0-rs)"
    r"|/Div_output_0-rs|output0-rs|output0_int8)$"
)
TAIL = (
    "/model/model.22/Concat_1_output_0-rs",
    "/model/model.22/Concat_1_output_0_sw",
    "/model/model.22/Sigmoid_output_0_rs",
    "output0-rs",
    "output0_int8",
)
RECIPES = ("h1", "h2", "h3", "h4", "h5", "h6", "c_all")
H5_TENSORS = 15


def select_layers(recipe: str, qp: dict) -> list[str]:
    """Return int8 tensor names from step-1 cfg kept in float16."""
    int8 = {k for k, v in qp.items() if v.get("dtype") == "int8"}
    layers: set[str] = set()
    if recipe == "h1":
        layers |= {k for k in int8 if H1.match(k)}
    if recipe in ("h2", "h4"):
        layers |= {k for k in int8 if H2.match(k)}
    if recipe in ("h5", "h6"):
        layers |= {k for k in int8 if BOX_TAIL.match(k)}
    if recipe == "h6":
        layers |= {k for k in int8 if H6B.match(k)}
    if recipe == "c_all":
        layers |= {k for k in int8 if "images" not in k}
    if recipe in ("h3", "h4"):
        assert all(t in qp and qp[t].get("dtype") == "int8" for t in TAIL), (
            "output tail names differ in this model"
        )
        layers |= set(TAIL)
    out = sorted(layers)
    assert out, "no layer matched"
    if recipe == "h5":
        assert len(out) == H5_TENSORS, (
            f"h5 expects {H5_TENSORS} tensors, matched {len(out)}"
        )
    return out


def build_hybrid(
    onnx_file: str | Path,
    dataset: str | Path,
    out_rknn: str | Path,
    recipe: str,
    *,
    batch: int = 1,
    target: str = "rk3588",
    verbose_file: str | Path | None = None,
) -> list[str]:
    """Build a hybrid INT8/FP16 RKNN model and return FP16 layer names.

    Quantizes with Ultralytics settings (mean 0, std 255, toolkit
    defaults), keeps the recipe tensors in float16, and exports .rknn.
    Working files of the toolkit go to a temporary directory.
    """
    from rknn.api import RKNN

    onnx_file, dataset, out_rknn = (
        Path(onnx_file).resolve(),
        Path(dataset).resolve(),
        Path(out_rknn).resolve(),
    )
    # verbose=True also prints the layer table that exporter logs keep
    kw = (
        {"verbose": True, "verbose_file": str(verbose_file)}
        if verbose_file
        else {"verbose": True}
    )
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            rknn = RKNN(**kw)
            assert (
                rknn.config(
                    mean_values=[[0, 0, 0]],
                    std_values=[[255, 255, 255]],
                    target_platform=target,
                )
                == 0
            )
            assert rknn.load_onnx(model=str(onnx_file)) == 0
            assert (
                rknn.hybrid_quantization_step1(
                    dataset=str(dataset), rknn_batch_size=batch
                )
                == 0
            )
            stem = onnx_file.stem
            cfg_file = Path(tmp) / f"{stem}.quantization.cfg"
            q = yaml.safe_load(cfg_file.read_text())
            layers = select_layers(recipe, q["quantize_parameters"])
            q["custom_quantize_layers"] = dict.fromkeys(layers, "float16")
            cfg_file.write_text(yaml.safe_dump(q, sort_keys=False))
            rknn.release()
            rknn = RKNN(**kw)
            assert (
                rknn.hybrid_quantization_step2(
                    model_input=f"{stem}.model",
                    data_input=f"{stem}.data",
                    model_quantization_cfg=f"{stem}.quantization.cfg",
                )
                == 0
            )
            assert rknn.export_rknn(str(out_rknn)) == 0
            rknn.release()
        finally:
            os.chdir(cwd)
    return layers
