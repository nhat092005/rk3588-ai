"""Tests for ai.npu.latency against the legacy-final scripts.

The fixtures are a slice of trace_c012.log with the outputs of the
legacy summarize.py and diff_cores.py on it (make_trace_fixtures.py).
"""

import pytest
import trace_cases

from ai.npu.latency import diff_cores, summarize, trace_log


def _expected(path):
    return path.read_text().splitlines()


def test_summarize_matches_legacy():
    got = summarize.summary_lines(trace_cases.TRACE_LOG)
    got[0] = "<log>"
    assert got == _expected(trace_cases.TRACE_SUMMARY)


def test_diff_cores_matches_legacy():
    got = diff_cores.diff_lines(trace_cases.TRACE_LOG)
    assert got == _expected(trace_cases.TRACE_DIFF)


def test_fixture_covers_every_kind():
    kinds = {
        summarize.KIND[t.en]
        for s in trace_log.parse_trace(trace_cases.TRACE_LOG)
        for t in s.tasks.get(2, [])
    }
    assert kinds == {"elementwise", "pool", "conv"}


def test_pool_tasks_differ_only_by_the_pc_address():
    pool = [
        line for line in _expected(trace_cases.TRACE_DIFF) if "en 0x60" in line
    ]
    assert pool
    assert all(line.endswith("ndiff 1") for line in pool)


@pytest.mark.parametrize(
    ("word", "expected"),
    [(0x10010000000E4004, (0x1001, 0xE, 0x4004))],
)
def test_split_word(word, expected):
    assert trace_log.split_word(word) == expected
