"""Prepare ppe5 dataset from data/css and data/ppe2286.

Builds 5 classes (Helmet, NoHelmet, NoVest, Vest, Person).
train/val/test come from CSS v28 (author split kept, 10 classes remapped
to 5, other classes dropped). test_ext is an external test set from
ppe2286: images where a COCO detector finds no person that lacks a head
label or a torso label, plus its Person boxes as pseudo-labels. See
data/ppe5/DATA_CARD.md. Images are relative symlinks, labels are new
files. raw/ of both sources is never modified.
Run from the rk3588-ai root:
    ../.venv/bin/python -m scripts.data.prepare_ppe5
"""

import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from ai.paths import DATA_DIR as DATA

ROOT = DATA / "ppe5"
PROCESSED = ROOT / "processed"
CSS_RAW = DATA / "css" / "raw"
PPE_RAW = DATA / "ppe2286" / "raw"

CLASSES = ["Helmet", "NoHelmet", "NoVest", "Vest", "Person"]
PERSON = 4
# CSS: Hardhat 0, NO-Hardhat 2, NO-Safety Vest 4, Person 5,
# Safety Vest 7 (see data/css/classes.txt)
CSS_REMAP = {0: 0, 2: 1, 4: 2, 5: 4, 7: 3}
CSS_SPLITS = {"train": "train", "valid": "val", "test": "test"}
HEAD_CLASSES, TORSO_CLASSES = (
    (0, 1),
    (2, 3),
)  # ppe2286 keeps its native indices

DETECTOR = "yolov8s.pt"  # COCO, class 0 = person
LOW_CONF = 0.1  # any detection from here up counts as a possible person
# Every such detection must reach HIGH_CONF to keep image and label.
HIGH_CONF = 0.6
BATCH = 32


def images_of(directory: Path) -> list[Path]:
    """Return sorted image files in a directory."""
    return sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def read_label_rows(path: Path) -> list[list[str]]:
    """Read and validate 5-column YOLO label rows from a file."""
    rows = [
        line.split() for line in path.read_text().splitlines() if line.strip()
    ]
    for row in rows:
        if len(row) != 5:
            raise ValueError(f"{path}: expected 5 fields per line, got {row}")
    return rows


def link_image(image: Path, dst_dir: Path) -> None:
    """Create a relative symlink to image inside dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    link = dst_dir / image.name
    if link.exists():
        raise FileExistsError(link)
    link.symlink_to(
        os.path.relpath(image.resolve(), dst_dir.resolve())
    )  # relative: no absolute path in the data


def write_label(rows: list[list[str]], image: Path, dst_dir: Path) -> None:
    """Write YOLO label rows for image into dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / f"{image.stem}.txt").write_text(
        "".join(" ".join(r) + "\n" for r in rows)
    )


def prepare_css() -> None:
    """Remap CSS v28 train/valid/test splits into PROCESSED."""
    for raw_split, split in CSS_SPLITS.items():
        images = images_of(CSS_RAW / raw_split / "images")
        for image in images:
            rows = read_label_rows(
                CSS_RAW / raw_split / "labels" / f"{image.stem}.txt"
            )
            kept = [
                [str(CSS_REMAP[int(r[0])]), *r[1:]]
                for r in rows
                if int(r[0]) in CSS_REMAP
            ]
            link_image(image, PROCESSED / "images" / split)
            write_label(kept, image, PROCESSED / "labels" / split)
        print(f"{split}: {len(images)} images")


def inside(x: float, y: float, box: list[float]) -> bool:
    """Return True if point (x, y) lies inside xyxy box."""
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def keep_with_persons(
    rows: list[list[str]], boxes: list[list[float]], confs: list[float]
) -> bool:
    """Check every detected person is confident and has head+torso."""
    head = [
        (float(r[1]), float(r[2])) for r in rows if int(r[0]) in HEAD_CLASSES
    ]
    torso = [
        (float(r[1]), float(r[2])) for r in rows if int(r[0]) in TORSO_CLASSES
    ]
    if (
        not head
        or not torso
        or not len(boxes)
        or min(confs) < HIGH_CONF
        or len(boxes) > min(len(head), len(torso))
    ):
        return False
    return all(
        any(inside(x, y, b) for x, y in head)
        and any(inside(x, y, b) for x, y in torso)
        for b in boxes
    )


def prepare_ext() -> None:
    """Filter ppe2286 images and generate test_ext split."""
    model = YOLO(DETECTOR)
    device = 0 if torch.cuda.is_available() else "cpu"
    items = [
        (split, image)
        for split in ("train", "valid")
        for image in images_of(PPE_RAW / split / "images")
    ]
    kept = 0
    for i in range(0, len(items), BATCH):
        batch = items[i : i + BATCH]
        results = model.predict(
            [str(p) for _, p in batch],
            classes=[0],
            conf=LOW_CONF,
            imgsz=640,
            device=device,
            verbose=False,
        )
        for (split, image), result in zip(batch, results, strict=True):
            rows = read_label_rows(
                PPE_RAW / split / "labels" / f"{image.stem}.txt"
            )
            boxes, confs = (
                result.boxes.xyxyn.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            )
            if not keep_with_persons(rows, boxes, confs):
                continue
            persons = [
                [str(PERSON), *(f"{v:.6f}" for v in xywh)]
                for xywh in result.boxes.xywhn.cpu().tolist()
            ]
            link_image(image, PROCESSED / "images" / "test_ext")
            write_label(
                rows + persons, image, PROCESSED / "labels" / "test_ext"
            )
            kept += 1
    print(f"test_ext: {kept} of {len(items)} ppe2286 images")


def write_dataset_yaml(
    dst: Path, classes: list[str], splits: dict[str, str]
) -> None:
    """Write dataset.yaml without a path key (root = yaml directory)."""
    lines = [f"{split}: {rel}" for split, rel in splits.items()]
    lines.append(f"nc: {len(classes)}")
    names = ", ".join(f'"{c}"' for c in classes)
    lines.append(f"names: [{names}]")
    dst.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Rebuild ppe5 processed dataset from CSS and ppe2286 raw data."""
    if PROCESSED.exists():
        # derived data only, rebuilt from the two raw/ directories
        shutil.rmtree(PROCESSED)
    prepare_css()
    prepare_ext()
    (ROOT / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    write_dataset_yaml(
        PROCESSED / "dataset.yaml",
        CLASSES,
        {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "test_ext": "images/test_ext",
        },
    )


if __name__ == "__main__":
    main()
