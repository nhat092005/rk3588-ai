"""ai.core must give exactly the outputs of legacy-final:ai/core (fixtures from tests/make_fixtures.py).

The full evaluation (evaluate() on css_test) is checked against the legacy leaderboard once the data is migrated.
"""

import cases
import numpy as np
import pytest

from ai.core import backends, evaluator, provenance
from rk3588_common import model_io

EXPECTED = np.load(cases.CORE_EXPECTED)


def test_dataset_fingerprint(tmp_path):
    ims, lbs = cases.write_dataset_files(tmp_path)
    assert evaluator.dataset_fingerprint(ims, lbs) == str(EXPECTED["fingerprint"])


@pytest.mark.parametrize("case", range(4))
def test_match_predictions(case):
    assert np.array_equal(evaluator.match_predictions(*cases.match_inputs(case)), EXPECTED[f"match{case}"])


def test_match_predictions_not_trivial():
    assert EXPECTED["match0"].any() and not EXPECTED["match0"].all()


@pytest.mark.parametrize(("quantize", "split"), [(8, False), (16, False), (8, True)])
def test_rknn_dump_backend(tmp_path, quantize, split):
    img = cases.write_dump(tmp_path, quantize, split)
    backend = backends.RknnDumpBackend(tmp_path / "dump.npz", tmp_path, evaluator.EVAL_CONF)
    got = backend(img, im_file="images/x.jpg").numpy()
    assert np.array_equal(got, EXPECTED[f"dump_q{quantize}_s{int(split)}"])


def test_rknn_dump_backend_rejects_other_input(tmp_path):
    img = cases.write_dump(tmp_path, 8, False)
    backend = backends.RknnDumpBackend(tmp_path / "dump.npz", tmp_path, evaluator.EVAL_CONF)
    with pytest.raises(ValueError, match="board input differs"):
        backend(img + 1, im_file="images/x.jpg")


def test_protocol_constants():
    got = [
        evaluator.PROTOCOL_VERSION,
        evaluator.IMGSZ,
        evaluator.EVAL_CONF,
        evaluator.EVAL_IOU,
        evaluator.EVAL_MAX_DET,
        evaluator.MAX_TIME_IMG,
    ]
    assert np.array_equal(np.array(got, dtype=str), EXPECTED["constants"])
    assert np.array_equal(evaluator.IOU_THRESHOLDS.numpy(), EXPECTED["iou_thresholds"])
    assert evaluator.PROTOCOL_VERSION is model_io.PROTOCOL_VERSION and evaluator.IMGSZ is model_io.IMGSZ


def test_eval_sets():
    got = [f"{k}={evaluator.EVAL_SETS[k]}" for k in ("css_test", "ppe2286_ext")]
    assert got == list(EXPECTED["eval_sets"])
    assert evaluator.EVAL_SETS_BY_DATASET == {"ppe5": list(EXPECTED["eval_sets_ppe5"])}


def test_code_repos():
    repos = provenance.code_repos()
    assert set(repos) == {"ai", "common"}
    assert all(set(state) == {"commit", "dirty"} for state in repos.values())
