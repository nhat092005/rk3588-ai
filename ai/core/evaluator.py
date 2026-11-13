"""Evaluation Protocol v1: one evaluator for every model and precision.

Dataset loading, LetterBox, NMS and AP come from the same Ultralytics code that
model.val() uses, so FP32 (PyTorch/ONNX) and NPU (RKNN FP16/INT8) results differ
only by the backend. Changing any constant below, or IMGSZ, requires a new PROTOCOL_VERSION
(both in rk3588_common.model_io).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG, nms
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywh2xyxy

from rk3588_common.model_io import IMGSZ, PROTOCOL_VERSION

from .dataset import dataset_yaml_path
from .provenance import code_repos

# Ultralytics val settings. EVAL_ names: the deployment NMS settings of rk3588_common.model_io are another contract.
EVAL_CONF = 0.001
EVAL_IOU = 0.7
EVAL_MAX_DET = 300
# Ultralytics NMS stops early after 2 + 0.05 s per image, which would silently truncate outputs with thousands of boxes
# above EVAL_CONF (hybrid recipes h3 and h4)
MAX_TIME_IMG = 60.0
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)

# eval set name -> (prepared dataset, split key in its dataset.yaml)
EVAL_SETS = {
    "css_test": ("ppe5", "test"),  # common test set for every model (82 images, CSS v28 test split)
    "ppe2286_ext": ("ppe5", "test_ext"),  # external test set (180 ppe2286 images, Person is a pseudo-label)
}
# Eval sets reported for a model trained on each dataset. ppe5_plus, ppe2286, ppe4 and their eval sets stay in
# legacy-final until those datasets migrate.
EVAL_SETS_BY_DATASET = {
    "ppe5": ["css_test", "ppe2286_ext"],
}

# A backend maps a uint8 RGB NCHW batch (1, 3, IMGSZ, IMGSZ) and its image path to raw predictions
# (1, 4 + nc, anchors) with xywh boxes in input pixels.
Backend = Callable[..., torch.Tensor]  # backend(img, im_file=...)


def build_eval_loader(eval_set: str):
    dataset_name, split = EVAL_SETS[eval_set]
    data_yaml = dataset_yaml_path(dataset_name)
    data = yaml.safe_load(data_yaml.read_text())
    if split not in data:
        raise KeyError(f"{data_yaml} has no '{split}' key; re-run scripts/data/prepare_{dataset_name}.py")
    img_path = str(Path(data["path"]) / data[split])
    data["names"] = dict(enumerate(data["names"]))
    data["channels"] = 3

    cfg = get_cfg(DEFAULT_CFG, {"imgsz": IMGSZ, "rect": False, "task": "detect", "mode": "val"})
    dataset = build_yolo_dataset(cfg, img_path, 1, data, mode="val", stride=32)
    loader = build_dataloader(dataset, batch=1, workers=4, shuffle=False)
    return loader, data["names"]


def dataset_fingerprint(im_files: list[str], label_files: list[str]) -> str:
    """sha256 over image bytes and label text, in file-name order.

    data/** is git-ignored, so this proves which data a number was measured on.
    """
    h = hashlib.sha256()
    for im, lb in sorted(zip(im_files, label_files, strict=True)):
        h.update(Path(im).name.encode())
        h.update(Path(im).read_bytes())
        h.update(Path(lb).read_bytes() if Path(lb).exists() else b"")
    return h.hexdigest()


def match_predictions(pred_cls: torch.Tensor, true_cls: torch.Tensor, iou: torch.Tensor) -> np.ndarray:
    """Greedy matching, same as BaseValidator.match_predictions (use_scipy=False)."""
    correct = np.zeros((pred_cls.shape[0], IOU_THRESHOLDS.numel()), dtype=bool)
    iou = (iou * (true_cls[:, None] == pred_cls)).cpu().numpy()
    for i, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def evaluate(backend: Backend, eval_set: str, limit: int | None = None) -> dict:
    """Run protocol v1 on an eval set and return a JSON-serializable result."""
    loader, names = build_eval_loader(eval_set)
    metrics = DetMetrics(names=names)
    im_files, label_files = [], []

    for n, batch in enumerate(loader):
        if limit is not None and n >= limit:
            break
        img = batch["img"]
        preds = backend(img, im_file=batch["im_file"][0])
        det = nms.non_max_suppression(
            preds,
            EVAL_CONF,
            EVAL_IOU,
            nc=0,
            multi_label=True,
            agnostic=False,
            max_det=EVAL_MAX_DET,
            max_time_img=MAX_TIME_IMG,
        )[0].cpu()

        cls = batch["cls"].squeeze(-1)
        gt = xywh2xyxy(batch["bboxes"]) * torch.tensor(img.shape[2:])[[1, 0, 1, 0]]
        pred_cls = det[:, 5]
        if cls.shape[0] and det.shape[0]:
            tp = match_predictions(pred_cls, cls, box_iou(gt, det[:, :4]))
        else:
            tp = np.zeros((det.shape[0], IOU_THRESHOLDS.numel()), dtype=bool)
        metrics.update_stats(
            {
                "tp": tp,
                "conf": det[:, 4].numpy(),
                "pred_cls": pred_cls.numpy(),
                "target_cls": cls.numpy(),
                "target_img": np.unique(cls.numpy()),
                "im_name": Path(batch["im_file"][0]).name,
            }
        )
        im_files.append(batch["im_file"][0])
        label_files.append(loader.dataset.label_files[n])  # shuffle=False, batch=1: batch n is item n

    metrics.process()
    box = metrics.box
    mp, mr, map50, map50_95 = box.mean_results()
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[names[int(c)]] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "ap50": round(float(ap50), 4),
            "ap50_95": round(float(ap), 4),
            "instances": int(metrics.nt_per_class[int(c)]),
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": {
            "input": f"LetterBox {IMGSZ}x{IMGSZ}, rect=False, scaleup=False, center, pad 114",
            "conf": EVAL_CONF,
            "iou": EVAL_IOU,
            "max_det": EVAL_MAX_DET,
            "multi_label": True,
        },
        "repos": code_repos(),
        "ultralytics_version": version("ultralytics"),
        "torch_version": version("torch"),
        "eval_set": eval_set,
        "dataset": EVAL_SETS[eval_set][0],
        "split": EVAL_SETS[eval_set][1],
        "n_images": len(im_files),
        "classes": list(names.values()),
        "dataset_fingerprint": dataset_fingerprint(im_files, label_files),
        "precision": round(float(mp), 4),
        "recall": round(float(mr), 4),
        "map50": round(float(map50), 4),
        "map50_95": round(float(map50_95), 4),
        "per_class": per_class,
    }
