"""Tests for ai.core.rknn against legacy-final outputs on real data."""

import json
import os
import subprocess
import sys

import cases
import cv2
import pytest

from ai.core.rknn import benchmark, calibration, hybrid

EXPECTED = json.loads(cases.RKNN_EXPECTED.read_text())
TEXT = cases.EVAL_PERF_TEXT.read_text()


def test_parse_layers():
    assert benchmark.parse_layers(TEXT) == EXPECTED["layers"]
    assert len(EXPECTED["layers"]) > 100


def test_parse_frequencies_and_total():
    assert benchmark.parse_frequencies(TEXT) == EXPECTED["frequencies"]
    assert (
        float(benchmark.LAYERS_TIME_RE.search(TEXT).group(1))
        == EXPECTED["layers_total_us"]
    )


def test_parse_summary_time():
    assert (
        benchmark.parse_summary_time_us("x\nTotal Time(us): 17219\n")
        == 17219.0
    )
    with pytest.raises(ValueError, match="Total Time"):
        benchmark.parse_summary_time_us(
            TEXT
        )  # perf_debug output has no summary line


def test_constants():
    assert sorted(benchmark.CORE_MASK) == EXPECTED["core_mask"]
    assert benchmark.PERF_REPEATS == EXPECTED["perf_repeats"]
    assert list(hybrid.RECIPES) == EXPECTED["recipes"]


@pytest.mark.parametrize("recipe", hybrid.RECIPES)
def test_select_layers(recipe):
    qp = json.loads(cases.STEP1_DTYPES.read_text())
    expected = EXPECTED["select_layers"][recipe]
    if isinstance(expected, str):
        with pytest.raises(AssertionError) as err:
            hybrid.select_layers(recipe, qp)
        assert f"AssertionError: {err.value}" == expected
    else:
        assert hybrid.select_layers(recipe, qp) == expected


def test_h5_matches_real_export():
    qp = json.loads(cases.STEP1_DTYPES.read_text())
    assert hybrid.select_layers("h5", qp) == EXPECTED["h5_export_record"]


def test_sample_calibration_images(tmp_path):
    labels = cases.write_calib_labels(tmp_path)
    assert (
        calibration.sample_calibration_images(
            labels, n=30, min_per_class=4, seed=0
        )
        == EXPECTED["calib_sample"]
    )
    assert (
        calibration.sample_calibration_images(labels)
        == EXPECTED["calib_sample_default"]
    )


def test_sample_calibration_images_ignores_hash_seed(tmp_path):
    """Sampler is independent of PYTHONHASHSEED."""
    labels = cases.write_calib_labels(tmp_path)
    code = (
        "import json, sys; from pathlib import Path; "
        "from ai.core.rknn.calibration import sample_calibration_images; "
        "print(json.dumps(sample_calibration_images("
        "Path(sys.argv[1]), n=30, min_per_class=4, seed=0)))"
    )
    lists = [
        json.loads(
            subprocess.check_output(
                [sys.executable, "-c", code, str(labels)],
                env={**os.environ, "PYTHONHASHSEED": s},
            )
        )
        for s in ("1", "2", "3")
    ]
    assert lists[0] == lists[1] == lists[2]
    assert len(lists[0]) == 30


def test_write_letterboxed_calib(tmp_path):
    srcs = cases.write_calib_images(tmp_path / "src")
    images = calibration.write_letterboxed_calib(srcs, tmp_path / "out")
    got = {
        p.stem: cases.digest(
            cv2.imread(str(images / f"{p.stem}.png"), cv2.IMREAD_UNCHANGED)
        )
        for p in srcs
    }
    assert got == EXPECTED["calib_png"]
