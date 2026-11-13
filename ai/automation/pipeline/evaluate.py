"""CLI: evaluate a run under Evaluation Protocol v1.

--backend pt/onnx run FP32 on the PC (GPU/CPU); rknn runs NPU inference
on the board (ai.board.probe.infer_dump, RKNNLite), since PC-side
rknn_server measured ~35s/image over Tailscale vs 42ms on-board.
Writes runs/<run_id>/eval/<backend>_<precision>_<eval_set>.json.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import yaml

from ai.core.backends import RknnDumpBackend, UltralyticsBackend
from ai.core.evaluator import (
    EVAL_SETS,
    EVAL_SETS_BY_DATASET,
    build_eval_loader,
    evaluate,
)
from ai.core.provenance import board_repo_args
from ai.paths import REPO_ROOT, RUNS_DIR
from rk3588_common.model_io import EVAL_CONF
from rk3588_common.provenance import file_sha256


def sh(cmd: list[str]) -> None:
    """Print and execute a shell command list."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def board_dump(
    run_dir: Path,
    precision: str,
    eval_set: str,
    board: str,
    board_dir: str,
    suffix: str = "",
) -> Path:
    """Copy model + images to board, run infer_dump, and copy dump back.

    The board holds this repo at <board_dir>/ai (meta-repo make
    sync-board) and its venv at <board_dir>/.venv.
    """
    board_repo = f"{board_dir}/ai"
    rknn_dir = run_dir / "weights" / f"rknn_{precision}{suffix}"
    rel_run = run_dir.relative_to(REPO_ROOT)
    loader, _ = build_eval_loader(eval_set)
    images = [
        str(Path(p).relative_to(REPO_ROOT)) for p in loader.dataset.im_files
    ]
    remote_dump = (
        f"{rel_run}/weights/npu_dump_{precision}{suffix}_{eval_set}.npz"
    )
    local_dump = (
        run_dir / "weights" / f"npu_dump_{precision}{suffix}_{eval_set}.npz"
    )
    remote_rknn = f"{rel_run}/weights/rknn_{precision}{suffix}"
    with tempfile.TemporaryDirectory() as tmp:
        lst = Path(tmp) / f"{eval_set}.txt"
        lst.write_text("\n".join(images) + "\n")
        sh(["ssh", board, f"mkdir -p {board_repo}/{rel_run}/weights"])
        sh(
            [
                "rsync",
                "-aL",
                f"--files-from={lst}",
                str(REPO_ROOT),
                f"{board}:{board_repo}/",
            ]
        )  # images (skips unchanged)
        sh(
            [
                "rsync",
                "-a",
                f"{rknn_dir}/",
                f"{board}:{board_repo}/{remote_rknn}/",
            ]
        )
        sh(
            [
                "scp",
                "-q",
                str(lst),
                f"{board}:{board_repo}/{rel_run}/weights/{lst.name}",
            ]
        )
    repo_args = " ".join(board_repo_args())
    sh(
        [
            "ssh",
            board,
            f"cd {board_repo} && "
            "../.venv/bin/python -m ai.board.probe.infer_dump "
            f"--model {remote_rknn} "
            f"--list {rel_run}/weights/{eval_set}.txt "
            f"--out {remote_dump} "
            f"--run-id {run_dir.name} {repo_args}",
        ]
    )
    sh(
        [
            "scp",
            "-q",
            f"{board}:{board_repo}/{remote_dump}",
            f"{board}:{board_repo}/{remote_dump[:-4]}.json",
            str(local_dump.parent),
        ]
    )
    return local_dump


def main() -> None:
    """Evaluate a run across eval sets and write JSON results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument(
        "--backend", choices=["pt", "onnx", "rknn"], required=True
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "int8", "int8h5"],
        help="rknn only (int8h5: hybrid INT8, float16 box-decode tail)",
    )
    parser.add_argument(
        "--board",
        help="rknn only: ssh target (BOARD of the meta-repo board.mk)",
    )
    parser.add_argument(
        "--board-dir",
        help="rknn only: board directory (BOARD_DIR of board.mk)",
    )
    parser.add_argument(
        "--device", default="0", help="pt only: CUDA index or cpu"
    )
    parser.add_argument(
        "--eval-set",
        choices=list(EVAL_SETS),
        help="default: every eval set of the run's dataset",
    )
    parser.add_argument(
        "--reuse-dump",
        action="store_true",
        help="rknn only: score local npu_dump again, no board access",
    )
    parser.add_argument(
        "--variant",
        default="",
        help=(
            "rknn only: quantization recipe name, evaluates "
            "weights/rknn_<precision>_<variant> and writes "
            "eval/rknn_<precision>_<variant>_<set>.json"
        ),
    )
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    dataset = yaml.safe_load((run_dir / "config.yaml").read_text())["dataset"]
    eval_sets = (
        [args.eval_set] if args.eval_set else EVAL_SETS_BY_DATASET[dataset]
    )
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)

    def save(
        result: dict, tag: str, eval_set: str, model_path: Path, extra: dict
    ) -> None:
        result = {
            **result,
            "run_id": args.run_id,
            "model_file": str(model_path.relative_to(run_dir)),
            "model_sha256": file_sha256(model_path),
            **extra,
        }
        path = out_dir / f"{tag}_{eval_set}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{path}: mAP50-95 {result['map50_95']}")

    if args.backend in ("pt", "onnx"):
        model_path = (
            run_dir
            / "weights"
            / ("best.pt" if args.backend == "pt" else "best.onnx")
        )
        backend = UltralyticsBackend(
            model_path, args.device if args.backend == "pt" else "cpu"
        )
        extra = {
            "backend": "pytorch"
            if args.backend == "pt"
            else "onnxruntime-cpu-pc"
        }
        for eval_set in eval_sets:
            save(
                evaluate(backend, eval_set),
                f"{args.backend}_fp32",
                eval_set,
                model_path,
                extra,
            )
        return

    if not args.precision:
        parser.error("--backend rknn needs --precision")
    if not args.reuse_dump and not (args.board and args.board_dir):
        parser.error(
            "--backend rknn needs --board and --board-dir (or --reuse-dump)"
        )
    suffix = f"_{args.variant}" if args.variant else ""
    rknn_dir = run_dir / "weights" / f"rknn_{args.precision}{suffix}"
    rknn_path = next(rknn_dir.glob("*.rknn"))
    for eval_set in eval_sets:
        dump = (
            run_dir
            / "weights"
            / f"npu_dump_{args.precision}{suffix}_{eval_set}.npz"
            if args.reuse_dump
            else board_dump(
                run_dir,
                args.precision,
                eval_set,
                args.board,
                args.board_dir,
                suffix,
            )
        )
        backend = RknnDumpBackend(dump, rknn_dir, EVAL_CONF)
        if backend.meta["model_sha256"] != file_sha256(rknn_path):
            raise ValueError("model on the board differs from the local .rknn")
        result = evaluate(backend, eval_set)
        if backend.rows:
            raise ValueError(
                f"{len(backend.rows)} board images were not evaluated"
            )
        save(
            result,
            f"rknn_{args.precision}{suffix}",
            eval_set,
            rknn_path,
            {
                "backend": "rknnlite-on-board",
                "sdk_version": backend.meta["sdk_version"],
                "board_dump": {
                    k: backend.meta[k]
                    for k in ("created_at", "kept_anchors", "normalized_boxes")
                },
            },
        )


if __name__ == "__main__":
    main()
