"""Inference backends for ai.core.evaluator.

Each backend takes a uint8 RGB NCHW batch (1, 3, 640, 640), already letterboxed
by the evaluator, and returns raw predictions (1, 4 + nc, anchors) with xywh
boxes in input pixels. NMS and metrics stay in the evaluator. The evaluator also passes
im_file, used by RknnDumpBackend to look up the board result of that image.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics.nn.autobackend import AutoBackend


class UltralyticsBackend:
    """PyTorch .pt or ONNX .onnx through Ultralytics AutoBackend, FP32, as model.val() runs them."""

    def __init__(self, weights: str | Path, device: str = "0"):
        dev = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.model = AutoBackend(str(weights), device=dev, fp16=False, fuse=True, verbose=False)
        self.model.eval()
        self.device = dev

    def __call__(self, img: torch.Tensor, im_file: str | None = None) -> torch.Tensor:
        with torch.no_grad():
            y = self.model(img.to(self.device).float() / 255)
        y = y[0] if isinstance(y, (list, tuple)) else y
        return y.float().cpu()


class RknnDumpBackend:
    """.rknn outputs computed on the board NPU by ai.board.probe.infer_dump (RKNNLite).

    The dump holds, per image, the md5 of the board's input and the raw output of every anchor whose
    best class score is > conf; this backend rebuilds the full output and refuses any image whose
    input differs from the evaluator's input.
    """

    def __init__(self, dump_path: str | Path, rknn_model_dir: str | Path, conf: float):
        dump_path = Path(dump_path)
        self.meta = json.loads(dump_path.with_suffix(".json").read_text())
        if self.meta["eval_conf"] != conf:
            raise ValueError(f"dump kept anchors above {self.meta['eval_conf']}, protocol conf is {conf}")
        # Ultralytics INT8 RKNN export normalizes boxes by input size (engine/exporter.py
        # _NormalizeCoords); nn/backends/rknn.py reverses it the same way.
        meta = yaml.safe_load((Path(rknn_model_dir) / "metadata.yaml").read_text())
        self.normalized = meta.get("args", {}).get("quantize") == 8 and not meta.get("split_head")
        with np.load(dump_path) as d:  # every d[key] access re-reads the array from the file: read once
            names, md5s, counts, idx, vals = (d[k] for k in ("names", "md5", "counts", "idx", "vals"))
            self.shape = tuple(int(x) for x in d["shape"])
        ends = np.cumsum(counts)
        self.rows = {
            str(n): (str(m), idx[e - c : e], vals[e - c : e])
            for n, m, c, e in zip(names, md5s, counts, ends, strict=True)
        }

    def __call__(self, img: torch.Tensor, im_file: str) -> torch.Tensor:
        name = Path(im_file).name
        if name not in self.rows:
            raise KeyError(f"{name} missing from the board dump")
        md5, idx, vals = self.rows.pop(name)
        if hashlib.md5(np.ascontiguousarray(img[0].permute(1, 2, 0).numpy()).tobytes()).hexdigest() != md5:
            raise ValueError(f"{name}: board input differs from the evaluator input")
        full = np.zeros(self.shape, dtype=np.float32)
        full[:, idx] = vals.T
        y = torch.from_numpy(full)[None]
        if self.normalized:
            h, w = img.shape[2:]
            y[:, [0, 2]] *= w
            y[:, [1, 3]] *= h
        return y
