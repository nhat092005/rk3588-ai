"""Pipeline plumbing tests: phase steps, dataset.yaml, logtee, args."""

import argparse
import json
import subprocess
import sys

import cases
import pytest
from ultralytics.data.utils import check_det_dataset

from ai.automation.pipeline import pipeline
from ai.core import dataset, provenance
from ai.paths import REPO_ROOT
from rk3588_common.provenance import add_provenance_args


@pytest.mark.parametrize("phase", ["phase2", "phase4"])
def test_phase_steps_match_legacy(phase):
    """Same make steps and done-files as legacy (board-sync renamed)."""
    expected = json.loads(cases.PIPELINE_EXPECTED.read_text())[phase]
    for step in expected:
        step[0] = [
            "sync-board" if part == "board-sync" else part for part in step[0]
        ]
    got = [
        [cmd, str(done.relative_to(REPO_ROOT)) if done else None]
        for cmd, done in pipeline.PHASES[phase]("ppe5")
    ]
    assert got == expected


def test_dataset_yaml_without_path(tmp_path, monkeypatch):
    """prepare_ppe5 writes no path key; yaml directory is root."""
    from scripts.data.prepare_ppe5 import write_dataset_yaml

    processed = tmp_path / "ppe5" / "processed"
    for split in ("train", "val"):
        (processed / "images" / split).mkdir(parents=True)
    write_dataset_yaml(
        processed / "dataset.yaml",
        ["A", "B"],
        {"train": "images/train", "val": "images/val"},
    )
    text = (processed / "dataset.yaml").read_text()
    assert "path" not in text
    assert str(tmp_path) not in text
    assert (
        check_det_dataset(str(processed / "dataset.yaml"))["path"]
        == processed.resolve()
    )
    monkeypatch.setattr(dataset, "DATA_DIR", tmp_path)
    data = dataset.load_dataset_yaml("ppe5")
    assert data["path"] == str(processed.resolve())
    assert data["names"] == ["A", "B"]


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
