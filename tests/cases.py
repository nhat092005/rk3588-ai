"""Deterministic inputs shared by make_fixtures.py (legacy code) and the ai tests."""

from pathlib import Path

import numpy as np
import yaml

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CORE_EXPECTED = FIXTURES / "core_expected.npz"


def write_dataset_files(root: Path) -> tuple[list[str], list[str]]:
    """Six images with labels, one image without a label file, names not in creation order."""
    rng = np.random.default_rng(10)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    ims, lbs = [], []
    for i, name in enumerate(["c", "a", "f", "b", "e", "d", "g"]):
        im, lb = root / "images" / f"{name}.jpg", root / "labels" / f"{name}.txt"
        im.write_bytes(rng.integers(0, 256, 300 + i, dtype=np.uint8).tobytes())
        if name != "g":
            lb.write_text(f"{i % 5} 0.5 0.5 0.{i + 1} 0.2\n")
        ims.append(str(im))
        lbs.append(str(lb))
    return ims, lbs


def match_inputs(case: int):
    """pred_cls (P,), true_cls (T,), iou (T, P) as torch tensors, with ties and several matches per target."""
    import torch

    rng = np.random.default_rng(20 + case)
    n_pred, n_true = [(40, 12), (5, 0), (0, 4), (60, 60)][case]
    pred_cls = torch.from_numpy(rng.integers(0, 5, n_pred).astype(np.float32))
    true_cls = torch.from_numpy(rng.integers(0, 5, n_true).astype(np.float32))
    iou = rng.choice([0.0, 0.45, 0.5, 0.62, 0.75, 0.75, 0.9, 0.95, 1.0], (n_true, n_pred)).astype(np.float32)
    return pred_cls, true_cls, torch.from_numpy(iou)


def write_dump(root: Path, quantize: int, split_head: bool):
    """A one-image RknnDumpBackend input: dump .npz + .json, metadata.yaml, and the matching evaluator image."""
    import hashlib

    import torch

    rng = np.random.default_rng(30 + quantize)
    img = torch.from_numpy(rng.integers(0, 256, (1, 3, 640, 640), dtype=np.uint8))
    md5 = hashlib.md5(np.ascontiguousarray(img[0].permute(1, 2, 0).numpy()).tobytes()).hexdigest()
    idx = np.sort(rng.choice(8400, 50, replace=False)).astype(np.int32)
    vals = rng.uniform(0, 1, (50, 9)).astype(np.float32)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / "dump.npz",
        names=np.array(["x.jpg"]),
        md5=np.array([md5]),
        counts=np.array([50]),
        idx=idx,
        vals=vals,
        shape=np.array([9, 8400]),
    )
    (root / "dump.json").write_text('{"eval_conf": 0.001}')
    meta = {"args": {"quantize": quantize}, **({"split_head": True} if split_head else {})}
    (root / "metadata.yaml").write_text(yaml.safe_dump(meta))
    return img
