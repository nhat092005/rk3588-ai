"""CLI entrypoint for standardized training, evaluation, and logging."""

import argparse
import csv
import importlib
import json
import shutil
from pathlib import Path

import yaml

from ai.core.backends import UltralyticsBackend
from ai.core.evaluator import EVAL_SETS_BY_DATASET
from ai.core.evaluator import evaluate as evaluate_v1
from ai.core.provenance import code_repos
from ai.core.trainer import TrainConfig, evaluate, train
from ai.paths import LEADERBOARDS_DIR, REPO_ROOT, RUNS_DIR
from rk3588_common.provenance import file_sha256


def sort_plots(source_dir: Path, plots_dir: Path, split_label: str) -> None:
    """Organize generated plot artifacts into directories by split."""
    stats_dir = plots_dir / "dataset_stats"
    samples_dir = plots_dir / "sample_predictions" / split_label
    curves_dir = plots_dir / "eval_curves" / split_label
    for d in (stats_dir, samples_dir, curves_dir):
        d.mkdir(parents=True, exist_ok=True)

    for f in list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")):
        if f.name.startswith(("labels.jpg", "labels_correlogram")):
            dest = stats_dir / f.name
        elif f.name.startswith(("train_batch", "val_batch")):
            dest = samples_dir / f.name
        else:
            dest = curves_dir / f.name
        shutil.move(str(f), str(dest))


def relativize_args(args_yaml: Path, root: Path = REPO_ROOT) -> None:
    """Make the repo paths in an Ultralytics args.yaml relative to root.

    Ultralytics writes data, project and save_dir as given (absolute).
    Tracked files hold no absolute paths (STANDARDS.md, section 4).
    """
    text = args_yaml.read_text()
    args_yaml.write_text(text.replace(f"{root}/", ""))


def append_leaderboard(dataset: str, row: dict) -> None:
    """Append an evaluation row to the dataset leaderboard CSV."""
    LEADERBOARDS_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADERBOARDS_DIR / f"{dataset}.csv"
    is_new = not path.exists()
    if not is_new:
        with path.open() as fh:
            header = next(csv.reader(fh))
        if header != list(row.keys()):
            raise ValueError(
                f"{path} header {header} differs from new row "
                f"{list(row.keys())}"
            )
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Train a model from config, evaluate, and update leaderboard."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--seed", type=int, help="overrides the config seed, e.g. 0"
    )
    parser.add_argument(
        "--print-run-id",
        action="store_true",
        help="print the run_id and exit (used by the Makefile for log paths)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    model_module = importlib.import_module(
        f"ai.models.{cfg['model']}.{cfg['model']}"
    )
    model_weights = model_module.WEIGHTS

    # No date: the same config + seed always maps to the same run
    run_id = (
        f"{cfg['dataset']}_{cfg['model']}_{cfg['epochs']}ep_s{cfg['seed']}"
    )
    # logs/ may already exist (created by the Makefile log wrapper);
    # a finished or started run has these
    if (RUNS_DIR / run_id / "weights").exists() or (
        RUNS_DIR / run_id / "config.yaml"
    ).exists():
        raise SystemExit(
            f"{RUNS_DIR / run_id} already exists; "
            "move it away or remove it before re-running."
        )
    if args.print_run_id:
        print(run_id)
        return

    train_cfg = TrainConfig(
        model=model_weights,
        dataset=cfg["dataset"],
        epochs=cfg["epochs"],
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        seed=cfg["seed"],
        device=cfg.get("device", "0"),
        optimizer=cfg.get("optimizer", "auto"),
        project=str(RUNS_DIR),
        run_name=run_id,
        pretrained=getattr(model_module, "PRETRAINED", None),
    )

    # snapshot now: training runs for hours on the code loaded here
    repos_at_start = code_repos()
    model = train(train_cfg)
    run_dir = RUNS_DIR / run_id

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        shutil.move(str(results_csv), run_dir / "metrics.csv")
    relativize_args(run_dir / "args.yaml")

    sort_plots(run_dir, run_dir / "plots", split_label="val")

    evaluate(
        model,
        cfg["dataset"],
        split="test",
        project=str(run_dir),
        run_name="test_eval",
    )
    test_eval_dir = run_dir / "test_eval"
    sort_plots(test_eval_dir, run_dir / "plots", split_label="test")
    shutil.rmtree(test_eval_dir, ignore_errors=True)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False)
    )  # seed as actually used
    (run_dir / "repos.json").write_text(
        json.dumps(repos_at_start, indent=2) + "\n"
    )

    # Leaderboard numbers come from the protocol v1 evaluator, not from
    # model.val() above (model.val() uses rect=True for .pt files; its
    # output is kept only for the plots).
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    backend = UltralyticsBackend(
        run_dir / "weights" / "best.pt", device=str(cfg.get("device", "0"))
    )
    for eval_set in EVAL_SETS_BY_DATASET[cfg["dataset"]]:
        result = {
            **evaluate_v1(backend, eval_set),
            "run_id": run_id,
            "backend": "pytorch",
            "model_file": "weights/best.pt",
            "model_sha256": file_sha256(run_dir / "weights" / "best.pt"),
        }
        (eval_dir / f"pt_fp32_{eval_set}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        append_leaderboard(
            cfg["dataset"],
            {
                "run_id": run_id,
                "model": cfg["model"],
                "epochs": cfg["epochs"],
                "seed": cfg["seed"],
                "eval_set": eval_set,
                "protocol": result["protocol_version"],
                "precision": result["precision"],
                "recall": result["recall"],
                "map50": result["map50"],
                "map50_95": result["map50_95"],
                # empty cell = class has no ground truth in eval set
                **{
                    f"ap50_95_{c}": result["per_class"]
                    .get(c, {})
                    .get("ap50_95", "")
                    for c in result["classes"]
                },
                "dataset_fingerprint": result["dataset_fingerprint"][:12],
                "ai_commit": result["repos"]["ai"]["commit"][:7],
                "common_commit": result["repos"]["common"]["commit"][:7],
            },
        )

    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
