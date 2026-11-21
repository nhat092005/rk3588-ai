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


def _word(target, reg, value):
    return f"{(target << 48) | (value << 16) | reg:016x}"


def _write_log(tmp_path, submits):
    """Write a log from a list of (mask, {slot: [(en, words)]})."""
    out = []
    for n, (mask, slots) in enumerate(submits):
        out.append(f"SUBMIT n={n} core_mask={mask} task_start=0 task_number=1")
        for slot, tasks in slots.items():
            for k, (en, words) in enumerate(tasks):
                out.append(
                    f"TASK sub={slot} idx={k} op=1 flags=0x0 en={en}"
                    f" intm=0x0 amount={len(words) - 4} offset=0 regcmd=0x0"
                )
                out.append("CMD " + " ".join(words))
    path = tmp_path / "trace.log"
    path.write_text("\n".join(out) + "\n")
    return path


def _pad(words):
    return [*words, *["0" * 16] * (4 - len(words))]


def _step(en, extra=0):
    return ("0x1", {0: [(en, _pad([_word(0x101, 0x10, extra)]))]})


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        (["0x18", "0x60", "0x1d"] * 3, 3),
        (["0x18", "0x60", "0x1d"], 3),
        (["0x18"] * 4, 1),
        (["0x18", "0x60", "0x18", "0x60", "0x18"], 5),
    ],
)
def test_inference_length(tmp_path, steps, expected):
    log = _write_log(tmp_path, [_step(en) for en in steps])
    assert summarize.inference_length(trace_log.parse_trace(log)) == expected


def test_unmapped_command_raises(tmp_path):
    log = _write_log(tmp_path, [_step("0x18")])
    log.write_text(log.read_text().replace("CMD ", "CMD unmapped "))
    with pytest.raises(ValueError, match="unmapped"):
        trace_log.parse_trace(log)


def test_height_uses_bits_12_to_0(tmp_path):
    height = [
        _word(0x1001, trace_log.DPU_DATA_CUBE_HEIGHT, 0xE000 | 26),
        *["0" * 16] * 3,
    ]
    task = [("0x1d", height)]
    trio = {2: task, 3: task, 4: task}
    log = _write_log(tmp_path, [("0x7", trio)])
    assert "(27, 27, 27): 1" in summarize.summary_lines(log)[2]


def test_unequal_task_counts_raise(tmp_path):
    cmd = ("0x1d", _pad([_word(0x101, 0x10, 0)]))
    log = _write_log(
        tmp_path, [("0x7", {2: [cmd, cmd], 3: [cmd, cmd], 4: [cmd]})]
    )
    with pytest.raises(ValueError, match="unequal"):
        diff_cores.diff_lines(log)
    with pytest.raises(ValueError, match="unequal"):
        summarize.summary_lines(log)


def test_length_difference_is_reported(tmp_path):
    short = ("0x1d", _pad([_word(0x101, 0x10, 0)]))
    long = ("0x1d", [*_pad([_word(0x101, 0x10, 0)]), _word(0x201, 0x1000, 1)])
    log = _write_log(tmp_path, [("0x7", {2: [short], 3: [long], 4: [short]})])
    assert diff_cores.diff_lines(log)[-1] == "    length differs: (4, 5, 4)"
