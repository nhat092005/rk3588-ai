"""Run a whole phase of Guide section 8 through make.

Each step = one make target + the result file proving it is done.
Finished steps are skipped, the first failure stops the phase, and
every phase ends with make tables.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from ai.core.evaluator import EVAL_SETS_BY_DATASET, PROTOCOL_VERSION
from ai.paths import LEADERBOARDS_DIR, REPO_ROOT, RUNS_DIR

AUTOMATION_DIR = Path(__file__).resolve().parents[1]
MODELS = [
    "yolov8n_baseline"
]  # yolov8s_baseline never had a ppe5 config, it stays in legacy-final
SEED = 0  # every phase trains one seed (Guide section 7)
MASKS_1CTX = ["AUTO", "CORE_0", "CORE_0_1", "CORE_0_1_2"]
MASKS_3CTX = "CORE_0,CORE_1,CORE_2"


def config_path(dataset: str, model: str) -> Path:
    """Return path to the training config YAML for dataset and model."""
    return AUTOMATION_DIR / "configs" / f"{dataset}_{model}.yaml"


def models_for(dataset: str) -> list[str]:
    """Return MODELS that have a config for this dataset."""
    return [m for m in MODELS if config_path(dataset, m).exists()]


def run_id(dataset: str, model: str, seed: int) -> str:
    """Resolve the deterministic run_id for dataset, model, and seed."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.automation.pipeline.run",
            str(config_path(dataset, model)),
            "--seed",
            str(seed),
            "--print-run-id",
        ],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if out.returncode == 0:
        return out.stdout.strip().splitlines()[-1]
    # run already exists: same naming rule as run.py
    import yaml

    cfg = yaml.safe_load(config_path(dataset, model).read_text())
    return f"{cfg['dataset']}_{cfg['model']}_{cfg['epochs']}ep_s{seed}"


def leaderboard_runs(dataset: str) -> list[str]:
    """Return sorted unique run_ids from the dataset leaderboard CSV."""
    path = LEADERBOARDS_DIR / f"{dataset}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return sorted(
            {
                r["run_id"]
                for r in csv.DictReader(fh)
                if r["protocol"] == PROTOCOL_VERSION
            }
        )


def step(
    target: str, done: Path | None, **make_vars: object
) -> tuple[list[str], Path | None]:
    """Build a (make command list, done file path) step tuple."""
    return ["make", target] + [f"{k}={v}" for k, v in make_vars.items()], done


def phase2(ds: str) -> list:
    """Build step list for phase 2 (training and evaluator check)."""
    first_set = EVAL_SETS_BY_DATASET[ds][0]
    steps = [
        train_step(ds, model, SEED, first_set) for model in models_for(ds)
    ]
    crosscheck_run = run_id(ds, MODELS[0], SEED)
    steps.append(
        step(
            "crosscheck",
            RUNS_DIR
            / crosscheck_run
            / "eval"
            / f"crosscheck_{first_set}.json",
            RUN=crosscheck_run,
        )
    )
    return steps


def train_step(
    ds: str, model: str, seed: int, first_set: str
) -> tuple[list[str], Path | None]:
    """Build a make train step tuple for dataset, model, and seed."""
    rid = run_id(ds, model, seed)
    return step(
        "train",
        RUNS_DIR / rid / "eval" / f"pt_fp32_{first_set}.json",
        CONFIG=config_path(ds, model).relative_to(REPO_ROOT),
        SEED=seed,
    )


def npu_eval_steps(rid: str, first_set: str, calib: str) -> list:
    """Build NPU export and evaluation steps for a run.

    calib is the run whose calibration list every INT8 export reuses
    (seed 0 of the first model), so INT8 and INT8H5 share one list.
    """
    rd = RUNS_DIR / rid
    return [
        step("export-npu", rd / "export_fp16.json", RUN=rid, PREC="fp16"),
        step(
            "export-npu",
            rd / "export_int8.json",
            RUN=rid,
            PREC="int8",
            CALIB_FROM=calib,
        ),
        step(
            "export-npu",
            rd / "export_int8h5.json",
            RUN=rid,
            PREC="int8h5",
            CALIB_FROM=calib,
        ),  # hybrid INT8 of phase 2c
        step(
            "evaluate",
            rd / "eval" / f"rknn_fp16_{first_set}.json",
            RUN=rid,
            BACKEND="rknn",
            PREC="fp16",
        ),
        step(
            "evaluate",
            rd / "eval" / f"rknn_int8_{first_set}.json",
            RUN=rid,
            BACKEND="rknn",
            PREC="int8",
        ),
        step(
            "evaluate",
            rd / "eval" / f"rknn_int8h5_{first_set}.json",
            RUN=rid,
            BACKEND="rknn",
            PREC="int8h5",
        ),
    ]


def phase3(ds: str) -> list:
    """Build step list for phase 3 (exports, complexity, and evals)."""
    first_set = EVAL_SETS_BY_DATASET[ds][0]
    steps = [
        step("sync-board", None)
    ]  # NPU accuracy runs on the board over ssh, no rknn_server needed
    for rid in leaderboard_runs(ds):
        rd = RUNS_DIR / rid
        npu = npu_eval_steps(rid, first_set, run_id(ds, MODELS[0], SEED))
        exports = [st for st in npu if st[0][1] == "export-npu"]
        steps += (
            exports
            + [
                step("complexity", rd / "complexity.json", RUN=rid),
                step(
                    "evaluate",
                    rd / "eval" / f"onnx_fp32_{first_set}.json",
                    RUN=rid,
                    BACKEND="onnx",
                ),
            ]
            + [st for st in npu if st[0][1] == "evaluate"]
        )
    return steps


def phase4(ds: str) -> list:
    """Build step list for phase 4 (tier-1, tier-2, throughput, GPU)."""
    steps = [step("board-check", None, LOCK=1), step("sync-board", None)]
    for model in models_for(ds):
        rid = run_id(ds, model, SEED)
        bench = RUNS_DIR / rid / "bench"
        steps.append(
            step(
                "benchmark-npu",
                bench / "tier1_fp16_AUTO.json",
                RUN=rid,
                PREC="fp16",
                CORE_MASK="AUTO",
            )
        )
        steps += [
            step(
                "benchmark-npu",
                bench / f"tier1_{p}_{m}.json",
                RUN=rid,
                PREC=p,
                CORE_MASK=m,
            )
            for p in ("int8", "int8h5")
            for m in MASKS_1CTX
        ]
        steps += [
            step(
                "bench-board",
                bench / f"tier2_{p}_AUTO.json",
                RUN=rid,
                PREC=p,
                CORE_MASK="AUTO",
            )
            for p in ("fp16", "int8", "int8h5")
        ]
        steps.append(
            step(
                "bench-board-cpu",
                bench / "tier2_cpu_onnx_t4.json",
                RUN=rid,
                THREADS=4,
            )
        )
        steps += [
            step(
                "bench-board-throughput",
                bench / f"throughput_int8_{m.replace(',', '-')}.json",
                RUN=rid,
                PREC="int8",
                CORE_MASKS=m,
            )
            for m in [*MASKS_1CTX, MASKS_3CTX]
        ]
        steps.append(step("bench-gpu", bench / "gpu_pt_fp32.json", RUN=rid))
    return steps


PHASES = {"phase2": phase2, "phase3": phase3, "phase4": phase4}


def main() -> None:
    """Execute all steps of the requested pipeline phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=list(PHASES))
    parser.add_argument(
        "--dataset", required=True, choices=list(EVAL_SETS_BY_DATASET)
    )
    args = parser.parse_args()

    steps = PHASES[args.phase](args.dataset)
    steps += [step("tables", None, DATASET=args.dataset)]
    for i, (cmd, done) in enumerate(steps, 1):
        label = f"[{args.phase} {i}/{len(steps)}] {' '.join(cmd)}"
        if done is not None and done.exists():
            print(
                f"{label}  SKIP (done: {done.relative_to(REPO_ROOT)})",
                flush=True,
            )
            continue
        print(label, flush=True)
        if subprocess.run(cmd, check=False, cwd=REPO_ROOT).returncode != 0:
            print(
                f"STOPPED at step {i}: fix the error above, then re-run "
                f"`make {args.phase} DATASET={args.dataset}`"
            )
            sys.exit(1)
        if done is not None and not done.exists():
            rel_done = done.relative_to(REPO_ROOT)
            print(
                f"STOPPED at step {i}: command succeeded "
                f"but {rel_done} was not written"
            )
            sys.exit(1)
    print(f"{args.phase} complete. Tables: results/{args.dataset}/")


if __name__ == "__main__":
    main()
