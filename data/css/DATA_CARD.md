# css — Data Card

- **Full name:** Construction Site Safety, Roboflow Universe Projects, **version 28** ("YOLOv5s"), exported
  2023-02-01 (`raw/README.roboflow.txt`), downloaded in YOLOv8 format.
- **Source:** https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety.
  **No dataset paper exists** as far as searched. Cite as a Roboflow Universe project (the Kaggle mirror suggests
  `@misc`, title "Construction Site Safety Dataset", author "Roboflow Universe Projects", Roboflow Universe, 2023).
- **License:** CC BY 4.0 (`raw/data.yaml`, `raw/README.dataset.txt`).
- **Image provenance (`raw/README.dataset.txt`):** frames from two YouTube videos (`Dhxf5mm7g1g`, `rYv9JZ2XBW4`)
  plus images cloned from Roboflow projects `personal-protective-equipment-combined-model`, `people-and-ladders`,
  `safety-vests`, `excavators-cwlh0`, `mit-indoor-scene-recognition` (null images), `people-detection-general`,
  `construction-madness`. Annotators are not stated.
- **Papers that train on it (neither states a version):** Ciftci, Turkdamar, Ozturk, JCTA 2024, section 4.1
  (https://publikasi.dinus.ac.id/index.php/jcta/article/view/10453); Deniz et al., arXiv 2409.15305, section III-C
  (https://arxiv.org/abs/2409.15305). Version 28 was chosen because it was the newest version on 2023-02-23, the
  access date in the arXiv paper (inference, not stated by the papers). `data.yaml` in
  `snehilsanyal/Construction-Site-Safety-PPE-Detection` also cites version 28.

## Scale and classes

- **2,801 images**: train 2,605 / valid 114 / test 82 (fixed Roboflow split). The current Roboflow page (v30,
  no augmentation) lists **717 original images** (521/114/82).
- Train is augmented 5x per source image (README: "5 versions of each source image"; about 514 distinct source
  names in train). Valid and test are not augmented. Augmentations: horizontal flip, crop 0 to 20%, rotation
  +-12 deg, shear +-2 deg, brightness +-25%, exposure +-20%, blur up to 0.5 px. Preprocessing: auto-orient,
  stretch to 640x640, "Modify Classes: 12 remapped, 4 dropped" (details not known).
- **10 classes** (`classes.txt`, identical to `raw/data.yaml` and the Kaggle class map):
  `0 Hardhat, 1 Mask, 2 NO-Hardhat, 3 NO-Mask, 4 NO-Safety Vest, 5 Person, 6 Safety Cone, 7 Safety Vest,
  8 machinery, 9 vehicle`.

Instances per split, five classes of interest (Helmet = Hardhat, NoHelmet = NO-Hardhat, Vest = Safety Vest,
NoVest = NO-Safety Vest):

| Class | Train (5x) | Valid | Test |
|---|---|---|---|
| Person | 9,532 | 166 | 174 |
| Hardhat | 3,145 | 79 | 110 |
| NO-Hardhat | 2,317 | 69 | 41 |
| Safety Vest | 3,033 | 41 | 61 |
| NO-Safety Vest | 3,962 | 106 | 90 |

## Verified checks (2026-10-25)

- v27 (`YOLOv8s`, 2023-01-10) and v28 have byte-identical valid and test labels. v27 train has 2,603 images
  (v28: 2,605), so v28 is used.
- Label meaning, from boxes drawn on valid/test images and crops: `NO-Hardhat` is a bare-head box,
  `NO-Safety Vest` is an upper-body box of a person without a vest, `Safety Vest` is mostly hi-vis vests and
  jackets (about 2 of 16 sampled crops look doubtful), `Hardhat` crops were all helmets.
- **Domain caveat:** about 8 of 12 sampled images are not construction scenes (selfies, a baby, a beach). They
  carry Person, NO-Hardhat and NO-Safety Vest labels. Small sample, drawn from images that contain NO-* classes.
- Label completeness on valid + test: of 143 images with a Person, 102 (71%) have person count not above both
  helmet-state labels (Hardhat + NO-Hardhat) and vest-state labels (Safety Vest + NO-Safety Vest).
- dHash (hamming 5 or less): no valid or test image is near a train image. Train images are augmented, so this
  check is not sensitive to every duplicate. 6 source names are shared by train and valid, 2 by train and test.
- Valid (114) and test (82) are small, so per-class metrics will vary a lot.

## Files

`raw/` is the original v28 export and is never modified. `processed/` has not been created yet.
