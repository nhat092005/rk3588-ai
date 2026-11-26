"""Pipeline plumbing tests: data.yaml, logtee, args."""

import argparse
import subprocess
import sys

import pytest

from ai.core import dataset, provenance
from ai.paths import REPO_ROOT
from rk3588_common.provenance import add_provenance_args


def test_dataset_yaml_without_path(tmp_path, monkeypatch):
    """data.yaml has no path key; its directory is the root."""
    root = tmp_path / "ds"
    root.mkdir()
    (root / "data.yaml").write_text(
        "nc: 2\nnames: [A, B]\ntrain: train/images\nval: valid/images\n"
    )
    monkeypatch.setattr(dataset, "DATA_DIR", tmp_path)
    data = dataset.load_dataset_yaml("ds")
    assert data["path"] == str(root.resolve())
    assert data["names"] == ["A", "B"]
    assert data["train"] == "train/images"
    with pytest.raises(FileNotFoundError, match=r"data\.yaml"):
        dataset.load_dataset_yaml("missing")


def test_logtee_header_and_exit_code(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/logtee.py",
            str(tmp_path / "t"),
            "--",
            sys.executable,
            "-c",
            "print('x\\ry'); raise SystemExit(3)",
        ],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3
    log = next(tmp_path.glob("t_*.log")).read_text()
    assert "# repos: ai=" in log
    assert ", common=" in log
    assert "\ny\n" in log
    assert "# exit_code: 3" in log


def test_board_repo_args_round_trip():
    parser = argparse.ArgumentParser()
    add_provenance_args(parser)
    args = parser.parse_args(["--run-id", "r", *provenance.board_repo_args()])
    assert dict(args.repos) == provenance.code_repos()


def test_relativize_args(tmp_path):
    """Repo paths in args.yaml become relative, other values stay."""
    from ai.automation.pipeline.run import relativize_args

    root = tmp_path / "rk3588-ai"
    args = tmp_path / "args.yaml"
    args.write_text(
        "task: detect\n"
        f"data: {root}/data/ds/data.yaml\n"
        "model: yolov8n.pt\n"
        f"project: {root}/experiments/runs\n"
        f"save_dir: {root}/experiments/runs/r0\n"
    )
    relativize_args(args, root)
    assert args.read_text() == (
        "task: detect\n"
        "data: data/ds/data.yaml\n"
        "model: yolov8n.pt\n"
        "project: experiments/runs\n"
        "save_dir: experiments/runs/r0\n"
    )
