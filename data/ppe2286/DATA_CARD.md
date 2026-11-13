# ppe2286 — Data Card

- **Full name:** Dataset of Personal Protective Equipment (dataset id `20250731-PPE2286`)
- **Source paper:** Huang M-L, Cheng Y. "Dataset of personal protective equipment." *Data in Brief*, 2025.
  DOI 10.1016/j.dib.2025.111988, full text PMC12398887 (https://pmc.ncbi.nlm.nih.gov/articles/PMC12398887/).
  Authors: National Chin-Yi University of Technology, Taichung, Taiwan.
- **Data repository (per the paper):** Mendeley Data, DOI 10.17632/zkzghjvpn2.6,
  https://data.mendeley.com/datasets/zkzghjvpn2/6.
- **Local copy:** downloaded manually as "Dataset of Personal Protective Equipment (PPE)" and copied unchanged
  into `raw/`. Its files are identical (`diff -rq`) to a Roboflow copy of the same dataset
  (https://universe.roboflow.com/ying2001/20250731-ppe2286-m0ega, uploaded by a Roboflow user). The
  repository host of the download was not checked beyond that match.
- **License:** conflicting statements. The article states CC BY-NC 4.0 (Europe PMC metadata and article text).
  `raw/README.dataset.txt` (Roboflow copy) states CC BY 4.0. Check before any redistribution.
- **Related article:** "SRR-YOLOv11" is listed in the paper as a working paper. No published version was found.

## Scale and classes

- **2,286 images**, 640x640 per the paper, no augmentation. Split: train 1,829 / valid 457 (80/20, random per the
  paper). **No test split.**
- **4 classes** (`classes.txt`, same as `raw/data.yaml`): `0 Helmet, 1 NoHelmet, 2 NoVest, 3 Vest`.
  **No Person class** (the paper's Limitations state it covers only these four categories).
- Image provenance (paper Table 1): 1,646 from GDUT-HWD (GitHub `wujixiu/helmet-detection`, Apache-2.0),
  134 from Roboflow `data-u4eek/helmetvest` (CC BY 4.0), 306 from Kaggle `niravnaik/safety-helmet-and-reflective-jacket`
  (Apache-2.0), 200 photographed by the authors.

Counts measured on `raw/*/labels` (train + valid):

| Class | Instances | Images containing the class |
|---|---|---|
| Helmet | 2,421 | 1,390 |
| NoHelmet | 1,174 | 622 |
| NoVest | 1,058 | 661 |
| Vest | 1,385 | 809 |

The paper's Table 2 ("The number of Images") reports the "images containing the class" column (1,130/260, 506/116,
521/140, 636/173 for train/valid), and its Table 3 ("The number of annotations") reports the instance column
(2,421, 1,174, 1,058, 1,385). Both match the counts above. Checked on 2026-10-26 against the full text (Europe PMC
XML of PMC12398887, saved at `tmp/ppe2286-paper/PMC12398887.xml`), section "Types of tags":

> "The data shown in Table 2 is not the total number of images, but the number of times each category appears in
> different images. For example, the Helmet category appears in 1130 training set images, which means that out of
> the 1829 training set images, 1130 images contain at least one Helmet."

## Verified checks (2026-10-25)

- Class order confirmed by drawing boxes on 12 random images (Helmet on helmets, Vest on vests, NoVest on torsos
  without vest, NoHelmet on bare heads) and by matching per-class image counts to the paper's Table 2.
- **Labels are sparse.** Only 907 of 2,286 images have both a head-state label (Helmet/NoHelmet) and a torso-state
  label (Vest/NoVest). The 134 `reflectivevest*` images have torso labels only. On those 907 images, a COCO
  detector (`yolov8s.pt`, conf 0.4) found no more persons than min(head labels, torso labels) in only 232 images,
  so many people carry no PPE label.
- **Class definitions** (paper, section "Types of tags"): "(1) NoHelmet: workers not wearing helmets". The paper gives
  no rule for improperly worn helmets, chin straps, caps or hoods. Labeling was manual in Roboflow Annotate, with
  "Double check and make corrections for any errors after the labeling is completed" (section "Experimental
  Design, Materials and Methods", Step 3). No inter-annotator agreement or error rate is reported.
- **Known label errors** (found 2026-10-26 by drawing boxes, both in `raw/train/`): `00002_jpg.rf.771be33b...` and
  `00055_jpg.rf.54135267...` have a NoHelmet box on a head that clearly wears a hard hat (red, yellow), which
  contradicts the definition above. The labels are unchanged in `raw/` and in `ppe5` `test_ext`. The error rate over
  the dataset is unknown (2 confirmed out of the 64 NoHelmet boxes in `ppe5` `test_ext`).
- GDUT-HWD is also a source of the CHV dataset, so overlap with CHV is likely (not measured).

## Files

`raw/` is the original download and is never modified. `raw/data.yaml` still holds the authors' Windows `path:`.
`raw/01_分割訓練及測試資料.py` is the authors' random 80/20 split script. In rk3588-ai only `raw/` is migrated (it
feeds `ppe5` `test_ext`); `prepare_ppe2286.py`, `processed/` and the eval sets `ppe2286_val` and `ppe2286_heldout`
stay in `legacy-final` until that line of work resumes (see `LEGACY.md`). In legacy, `processed/` is built by `scripts/data/prepare_ppe2286.py` (`make prepare-data DATASET=ppe2286`): symlinks to
the authors' `train` (1,829) and `valid` (457), 4 classes as published, no filtering. The authors give no test
split (`raw/data.yaml` has an empty `test:`), so `processed/dataset.yaml` sets `test` to the same images as `val`.
Eval set `ppe2286_val` therefore measures validation images that also select `best.pt`, and is optimistic. Labels are
sparse (see above), so missing boxes count as false positives and lower precision.

`processed/` also has a `heldout` split: the 457 valid images minus the 50 that are in `ppe5` `test_ext` (so 407).
Neither `ppe5` nor `ppe5_plus` trains on them (`ppe5_plus` trains on the 180 `test_ext` images, 130 from `train` and
50 from `valid`). Eval set `ppe2286_heldout` is therefore a ppe2286-domain test set for models trained on `ppe5`,
`ppe5_plus` and `ppe4`. A model trained on `ppe2286` has not trained on these images either, but its `best.pt` was
selected on `val`, which contains them, so its score here is slightly optimistic. The script needs `data/ppe5/processed`
(it reads the `test_ext` names) and asserts that no heldout image is in a ppe5 train, val or test split.

Overlap check of `heldout` (407 images) with dHash (9x8, hamming distance), measured 2026-10-28. Images of `heldout`
with a match at hamming 5 or less, and at 8 or less:

| Compared with | Images | At most 5 | At most 8 |
|---|---|---|---|
| `ppe5` `test_ext` (the 180 images `ppe5_plus` adds to train) | 180 | 0 | 0 |
| `ppe5` train / val / test (CSS v28) | 2,605 / 114 / 82 | 0 / 0 / 0 | 0 / 0 / 0 |
| `ppe2286` `train` | 1,829 | 5 | 8 |

So the `ppe5_plus` gain on `heldout` is not caused by near-copies of its extra training images. 8 of 407 `heldout`
images (2%) are near-copies of ppe2286 training images, which adds slightly to the optimism of the `ppe2286` model's
score on this set (effect size not measured, the 8 pairs were not inspected). dHash finds near-identical or resized
images only, not the same scene in another frame or crop. The thresholds (5 and 8) are the ones used for the CSS vs
ppe2286 check, not a standard.
