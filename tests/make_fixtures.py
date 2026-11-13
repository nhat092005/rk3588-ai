"""Write the expected outputs of ai.core from the legacy code at tag legacy-final.

Run once with the legacy interpreter from this repo's root, never imported by the tests:
    <legacy>/.venv/bin/python tests/make_fixtures.py <legacy>
<legacy> is the legacy checkout (HEAD at legacy-final, ai/core clean).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cases
import numpy as np


def check_legacy(legacy: Path) -> None:
    tag = subprocess.check_output(["git", "-C", str(legacy), "describe", "--exact-match", "--tags", "HEAD"], text=True)
    assert tag.strip() == "legacy-final", f"legacy HEAD is at {tag.strip()}, not legacy-final"
    dirty = subprocess.check_output(["git", "-C", str(legacy), "status", "--porcelain", "--", "ai/core"], text=True)
    assert not dirty, f"legacy ai/core has local changes:\n{dirty}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    legacy = parser.parse_args().legacy.resolve()
    check_legacy(legacy)
    sys.path.insert(0, str(legacy))
    from ai.core import backends as old_backends  # legacy-final:ai/core/backends.py
    from ai.core import evaluator as old  # legacy-final:ai/core/evaluator.py

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        ims, lbs = cases.write_dataset_files(Path(tmp))
        out["fingerprint"] = np.array(old.dataset_fingerprint(ims, lbs))
    for case in range(4):
        out[f"match{case}"] = old.match_predictions(*cases.match_inputs(case))
    for quantize, split in ((8, False), (16, False), (8, True)):
        with tempfile.TemporaryDirectory() as tmp:
            img = cases.write_dump(Path(tmp), quantize, split)
            backend = old_backends.RknnDumpBackend(Path(tmp) / "dump.npz", tmp, old.CONF)
            out[f"dump_q{quantize}_s{int(split)}"] = backend(img, im_file="images/x.jpg").numpy()
    out["eval_sets"] = np.array([f"{k}={old.EVAL_SETS[k]}" for k in ("css_test", "ppe2286_ext")])
    out["eval_sets_ppe5"] = np.array(old.EVAL_SETS_BY_DATASET["ppe5"])
    out["constants"] = np.array(
        [old.PROTOCOL_VERSION, old.IMGSZ, old.CONF, old.IOU, old.MAX_DET, old.MAX_TIME_IMG], dtype=str
    )
    out["iou_thresholds"] = old.IOU_THRESHOLDS.numpy()
    cases.FIXTURES.mkdir(exist_ok=True)
    np.savez_compressed(cases.CORE_EXPECTED, **out)
    print(f"wrote {cases.CORE_EXPECTED.name} ({len(out)} arrays)")


if __name__ == "__main__":
    main()
