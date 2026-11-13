"""Lock Evaluation Protocol v1: golden test and cross-check against model.val().

golden: evaluator must reproduce golden/evaluator_v1.json (--update rewrites it).
crosscheck: evaluator vs model.val(rect=False); .pt files default to rect=True, but the
NPU needs a fixed 640x640 input. Golden model: public yolov8n.pt, pinned by sha256.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

from ai.core.backends import UltralyticsBackend
from ai.core.dataset import load_dataset_yaml
from ai.core.evaluator import EVAL_SETS, PROTOCOL_VERSION, evaluate
from ai.core.provenance import code_repos
from ai.paths import GOLDEN_DIR, RUNS_DIR
from rk3588_common.provenance import provenance

GOLDEN_PATH = GOLDEN_DIR / f"evaluator_{PROTOCOL_VERSION}.json"
GOLDEN_WEIGHTS = "yolov8n.pt"
GOLDEN_EVAL_SET = "css_test"
GOLDEN_DEVICE = "cpu"  # CPU kernels are deterministic; GPU results may differ in the 4th decimal between machines
TOLERANCE = 5e-4  # max |difference| allowed on any metric
CROSSCHECK_TOLERANCE = 1e-3


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_items(result: dict) -> dict[str, float]:
    items = {k: result[k] for k in ("precision", "recall", "map50", "map50_95")}
    for name, row in result["per_class"].items():
        for k in ("precision", "recall", "ap50", "ap50_95"):
            items[f"{name}.{k}"] = row[k]
    return items


def compare(expected: dict, actual: dict, tol: float) -> list[str]:
    exp, act = metric_items(expected), metric_items(actual)
    errors = [f"{k}: missing" for k in exp if k not in act]
    errors += [f"{k}: expected {exp[k]}, got {act[k]}" for k in exp if k in act and abs(exp[k] - act[k]) > tol]
    return errors


def golden(update: bool) -> int:
    weights = YOLO(GOLDEN_WEIGHTS).ckpt_path  # downloads yolov8n.pt if missing
    result = evaluate(UltralyticsBackend(weights, device=GOLDEN_DEVICE), GOLDEN_EVAL_SET)
    result["golden_weights"] = GOLDEN_WEIGHTS
    result["golden_weights_sha256"] = sha256(weights)

    if update:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Golden file written: {GOLDEN_PATH}")
        return 0

    expected = json.loads(GOLDEN_PATH.read_text())
    errors = []
    for key in ("protocol_version", "protocol", "n_images", "dataset_fingerprint", "golden_weights_sha256"):
        if expected[key] != result[key]:
            errors.append(f"{key}: expected {expected[key]}, got {result[key]}")
    errors += compare(expected, result, TOLERANCE)
    if errors:
        print("GOLDEN TEST FAILED\n" + "\n".join(errors))
        return 1
    print(f"GOLDEN TEST PASSED ({len(metric_items(result))} metrics within {TOLERANCE})")
    return 0


def crosscheck(weights: str, eval_set: str, device: str, out: Path | None = None) -> int:
    ours = evaluate(UltralyticsBackend(weights, device=device), eval_set)

    dataset_name, split = EVAL_SETS[eval_set]
    data = load_dataset_yaml(dataset_name)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_yaml = Path(tmp) / "data.yaml"
        tmp_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": data["path"],
                    "train": data["train"],
                    "val": data["val"],
                    "test": str(Path(data["path"]) / data[split]),
                    "names": data["names"],
                }
            )
        )
        box = (
            YOLO(weights)
            .val(
                data=str(tmp_yaml),
                split="test",
                imgsz=640,
                batch=1,
                rect=False,
                device=device,
                plots=False,
                project=tmp,
                name="val",
                verbose=False,
            )
            .box
        )

    names = data["names"]
    theirs = {
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "per_class": {
            names[int(c)]: {
                k: round(float(v), 4)
                for k, v in zip(("precision", "recall", "ap50", "ap50_95"), box.class_result(i), strict=True)
            }
            for i, c in enumerate(box.ap_class_index)
        },
    }
    errors = compare(theirs, ours, CROSSCHECK_TOLERANCE)
    diffs = {k: round(abs(v - metric_items(theirs).get(k, v)), 4) for k, v in metric_items(ours).items()}
    report = {
        **provenance(code_repos()),
        "weights": str(weights),
        "eval_set": eval_set,
        "tolerance": CROSSCHECK_TOLERANCE,
        "passed": not errors,
        "max_abs_diff": max(diffs.values()),
        "evaluator": metric_items(ours),
        "model_val_rect_false": metric_items(theirs),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if out is not None and not errors:  # only a passing check marks the step as done
        out.write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        print("CROSSCHECK FAILED\n" + "\n".join(errors))
        return 1
    print(f"CROSSCHECK PASSED ({len(metric_items(ours))} metrics within {CROSSCHECK_TOLERANCE})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden")
    g.add_argument("--update", action="store_true")
    c = sub.add_parser("crosscheck")
    c.add_argument("--weights", default=GOLDEN_WEIGHTS)
    c.add_argument("--run-id", help="use runs/<id>/weights/best.pt and save runs/<id>/eval/crosscheck_<eval_set>.json")
    c.add_argument("--eval-set", default=GOLDEN_EVAL_SET, choices=list(EVAL_SETS))
    c.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.cmd == "golden":
        sys.exit(golden(args.update))
    out = None
    if args.run_id:
        run_dir = RUNS_DIR / args.run_id
        args.weights, out = str(run_dir / "weights" / "best.pt"), run_dir / "eval" / f"crosscheck_{args.eval_set}.json"
    sys.exit(crosscheck(args.weights, args.eval_set, args.device, out))


if __name__ == "__main__":
    main()
