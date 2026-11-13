"""Calibration image selection utilities for RKNN INT8 quantization."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path


def sample_calibration_images(
    labels_dir: Path,
    n: int = 100,
    min_per_class: int = 10,
    seed: int = 0,
) -> list[str]:
    """Select calibration image stems with class-coverage guarantees."""
    rng = random.Random(seed)
    class_to_images: dict[int, set[str]] = defaultdict(set)
    all_images: list[str] = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        classes = {
            int(line.split()[0])
            for line in label_file.read_text().splitlines()
            if line.strip()
        }
        all_images.append(label_file.stem)
        for c in classes:
            class_to_images[c].add(label_file.stem)

    selected: set[str] = set()
    for c in sorted(class_to_images, key=lambda c: len(class_to_images[c])):
        # Set order depends on PYTHONHASHSEED; sort before shuffle.
        candidates = sorted(class_to_images[c] - selected)
        rng.shuffle(candidates)
        selected.update(candidates[:min_per_class])

    remaining = [img for img in all_images if img not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: max(0, n - len(selected))])

    return sorted(selected)[:n]


def write_letterboxed_calib(image_paths: list[Path], out_dir: Path) -> Path:
    """Write calibration images letterboxed to IMGSZ x IMGSZ as PNG.

    RKNN-Toolkit2 resizes calibration images that differ from the input
    size in an undocumented way, so they are preprocessed here like the
    evaluator and board (rk3588_common.model_io.preprocess) and stored
    lossless. Returns the images directory.
    """
    import cv2

    from rk3588_common.model_io import preprocess

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for src in image_paths:
        rgb = preprocess(cv2.imread(str(src)))[0][0]
        cv2.imwrite(
            str(images_dir / f"{src.stem}.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
    return images_dir
