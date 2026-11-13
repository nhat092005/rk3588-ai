"""Deterministic inputs shared by make_fixtures.py and ai tests."""

from pathlib import Path

import numpy as np
import yaml

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CORE_EXPECTED = FIXTURES / "core_expected.npz"
RKNN_EXPECTED = FIXTURES / "rknn_expected.json"
# Real inputs copied from legacy-final by make_fixtures.py: an
# eval_perf(perf_debug=True) output of the baseline INT8 model on
# CORE_0, and the tensor dtypes of the step-1 quantization cfg of the
# SiLU baseline (hybrid recipes).
EVAL_PERF_TEXT = FIXTURES / "eval_perf_debug_int8_core0.txt"
STEP1_DTYPES = FIXTURES / "step1_silu_dtypes.json"
CALIB_SHAPES = [
    (640, 640),
    (480, 640),
    (1080, 1920),
    (200, 300),
    (641, 639),
    (429, 640),
    (640, 429),
    (3, 1000),
]


def write_dataset_files(root: Path) -> tuple[list[str], list[str]]:
    """Six labeled images, one unlabeled image, unordered creation."""
    rng = np.random.default_rng(10)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    ims, lbs = [], []
    for i, name in enumerate(["c", "a", "f", "b", "e", "d", "g"]):
        im, lb = (
            root / "images" / f"{name}.jpg",
            root / "labels" / f"{name}.txt",
        )
        im.write_bytes(rng.integers(0, 256, 300 + i, dtype=np.uint8).tobytes())
        if name != "g":
            lb.write_text(f"{i % 5} 0.5 0.5 0.{i + 1} 0.2\n")
        ims.append(str(im))
        lbs.append(str(lb))
    return ims, lbs


def match_inputs(case: int):
    """pred_cls (P,), true_cls (T,), iou (T, P) as torch tensors."""
    import torch

    rng = np.random.default_rng(20 + case)
    n_pred, n_true = [(40, 12), (5, 0), (0, 4), (60, 60)][case]
    pred_cls = torch.from_numpy(rng.integers(0, 5, n_pred).astype(np.float32))
    true_cls = torch.from_numpy(rng.integers(0, 5, n_true).astype(np.float32))
    iou = rng.choice(
        [0.0, 0.45, 0.5, 0.62, 0.75, 0.75, 0.9, 0.95, 1.0], (n_true, n_pred)
    ).astype(np.float32)
    return pred_cls, true_cls, torch.from_numpy(iou)


def write_dump(root: Path, quantize: int, split_head: bool):
    """One-image RknnDumpBackend input: dump.npz/json + metadata."""
    import hashlib

    import torch

    rng = np.random.default_rng(30 + quantize)
    img = torch.from_numpy(
        rng.integers(0, 256, (1, 3, 640, 640), dtype=np.uint8)
    )
    md5 = hashlib.md5(
        np.ascontiguousarray(img[0].permute(1, 2, 0).numpy()).tobytes()
    ).hexdigest()
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
    meta = {
        "args": {"quantize": quantize},
        **({"split_head": True} if split_head else {}),
    }
    (root / "metadata.yaml").write_text(yaml.safe_dump(meta))
    return img


def digest(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(
        f"{a.dtype}{a.shape}".encode() + np.ascontiguousarray(a).tobytes()
    ).hexdigest()


def write_calib_labels(root: Path) -> Path:
    """80 label files over 5 classes, two objects each."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(80):
        (root / f"img{i:03d}.txt").write_text(
            f"{i % 5} 0.5 0.5 0.1 0.1\n{(i * 3) % 5} 0.2 0.2 0.1 0.1\n"
        )
    return root


def write_calib_images(root: Path) -> list[Path]:
    """PNG sources (lossless) of CALIB_SHAPES."""
    import cv2

    rng = np.random.default_rng(40)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (h, w) in enumerate(CALIB_SHAPES):
        path = root / f"src{i}.png"
        cv2.imwrite(str(path), rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        paths.append(path)
    return paths
