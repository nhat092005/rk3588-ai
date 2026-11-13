# ppe5 — Data Card

Derived dataset, 5 classes: `0 Helmet, 1 NoHelmet, 2 NoVest, 3 Vest, 4 Person` (`classes.txt`).
Built by `scripts/data/prepare_ppe5.py` (`make prepare-data DATASET=ppe5`) from two sources. `processed/` holds
symlinked images and new label files only. `raw/` of the sources is never modified.

| Split | Source | Images | Notes |
|---|---|---|---|
| `train` | `data/css/raw/train` | 2,605 | CSS v28 author split, 5x augmented from about 514 source names |
| `val` | `data/css/raw/valid` | 114 | |
| `test` | `data/css/raw/test` | 82 | eval set `css_test` |
| `test_ext` | `data/ppe2286/raw` (train + valid) | 180 | external test, never used for training, eval set `ppe2286_ext` |

## Sources

- **CSS v28:** Roboflow Universe Projects, Construction Site Safety, CC BY 4.0, no dataset paper. See `data/css/DATA_CARD.md`.
- **ppe2286:** Huang and Cheng, *Data in Brief* 2025, DOI 10.1016/j.dib.2025.111988. See `data/ppe2286/DATA_CARD.md`.

## Class mapping

CSS (10 classes) to ppe5:

| CSS class (index) | ppe5 class (index) |
|---|---|
| Hardhat (0) | Helmet (0) |
| NO-Hardhat (2) | NoHelmet (1) |
| NO-Safety Vest (4) | NoVest (2) |
| Safety Vest (7) | Vest (3) |
| Person (5) | Person (4) |
| Mask (1), NO-Mask (3), Safety Cone (6), machinery (8), vehicle (9) | dropped (objects removed from labels, images kept) |

ppe2286 keeps its native indices 0 to 3 (Helmet, NoHelmet, NoVest, Vest). Person (4) is added by the script.

## test_ext selection and Person pseudo-labels

`test_ext` is **not** a human-labeled Person set. For each of the 2,286 ppe2286 images, the script runs the COCO
detector `yolov8s.pt` (class 0 only, imgsz 640, sha256 prefix `1f47a78bf100391c`) and keeps an image only if:

1. it has at least one head-state label (Helmet or NoHelmet) and one torso-state label (NoVest or Vest),
2. at least one detection exists at conf 0.1 or above, and every such detection has conf 0.6 or above,
3. the number of detections is at most min(head labels, torso labels),
4. every detected person box contains the centre of at least one head label and one torso label.

Kept images get the detections as Person boxes (conf 0.6 or above). 180 of 2,286 images pass. A looser rule (conf 0.4,
count only) would keep 232, so 52 images are dropped by the two-threshold rule. Only the first of the five review
pages of those 52 was inspected by eye. In that part many were good images rejected because of a tiny extra detection,
so the rule is conservative. The other pages were not inspected. Person AP on `test_ext` measures agreement with a COCO
detector, not with human labels. Report the four PPE classes as the main result and Person separately with this note.

## Measured counts (2026-10-25, instances per class)

| Split | Helmet | NoHelmet | NoVest | Vest | Person |
|---|---|---|---|---|---|
| train | 3,145 | 2,317 | 3,962 | 3,033 | 9,532 |
| val | 79 | 69 | 106 | 41 | 166 |
| test | 110 | 41 | 90 | 61 | 174 |
| test_ext | 184 | 64 | 104 | 144 | 247 |

Empty label files (background after dropping classes): train 70, val 30, test 22, test_ext 0.

## Known limits

- `train` is 5x augmented from about 514 source images, so the real number of distinct training images is far
  smaller than 2,605. Valid and test are not augmented.
- `val` and `test` are small (114 and 82 images), per-class metrics vary a lot. `test` has only 41 NoHelmet instances.
- Many CSS images are not construction scenes (selfies, a baby, a beach), they carry Person, NoHelmet and NoVest labels.
- `test_ext` covers scenes of the "clean" kind (mostly stock-style photos), not crowded or distant surveillance scenes.
- Person labels in `test_ext` are pseudo-labels. A COCO detector can still miss occluded or very small people.
- Selection ran the detector on GPU. A CPU run may differ on borderline images. The 180 list was confirmed identical
  to the earlier analysis run (`tmp/ppe2286-person-check/passing_180.txt`).
- CSS v28 and ppe2286 were checked for overlap with dHash (hamming 5 or less): 5 ppe2286 images are near-duplicates
  of CSS `valid` images (none of CSS `train`, but train copies are augmented so the check is less sensitive there).
  One of them, `02912_jpg.rf.d68699bf...`, is in `test_ext`. At hamming 8 or less, 13 images match and a second one,
  `helmet_jacket_00482_jpg.rf.eb97e695...`, is in `test_ext`. They were kept, 2 of 180 images.
- Two `test_ext` images carry a NoHelmet label on a head that wears a hard hat (`00002_jpg.rf.771be33b...`,
  `00055_jpg.rf.54135267...`), a label error of ppe2286 (see `data/ppe2286/DATA_CARD.md`). They were kept, so
  NoHelmet on `test_ext` is slightly underestimated. The model also fails on more NoHelmet boxes than these two:
  on the default YOLOv8n run, 17 of 64 NoHelmet boxes had no head detection at conf 0.25 (not yet inspected).
