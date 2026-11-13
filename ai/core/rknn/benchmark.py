"""Tier-1 measurements of an RKNN model on the board, driven from the PC.

Timing comes from board hardware via eval_perf(), unaffected by PC-board network delay.
Needs adbd + rknn_server on the board (rk3588-deploy sh/start_rknn_server.sh).
Output formats match RKNN-Toolkit2 2.3.2, tested in tests/test_rknn.py on a real eval_perf output.
"""

from __future__ import annotations

import re
import statistics
from contextlib import contextmanager

from rknn.api import RKNN

CORE_MASK = {
    "AUTO": RKNN.NPU_CORE_AUTO,
    "CORE_0": RKNN.NPU_CORE_0,
    "CORE_1": RKNN.NPU_CORE_1,
    "CORE_2": RKNN.NPU_CORE_2,
    "CORE_0_1": RKNN.NPU_CORE_0_1,
    "CORE_0_1_2": RKNN.NPU_CORE_0_1_2,
}
PERF_REPEATS = 5  # eval_perf calls per latency session; latency = median

# perf_debug=False: "Total Time(us): 17219"
SUMMARY_TIME_RE = re.compile(r"^Total Time\(us\):\s*([\d.]+)", re.M)
# perf_debug=True: "Total Operator Elapsed Per Frame Time(us): 20663"
LAYERS_TIME_RE = re.compile(r"^Total Operator Elapsed Per Frame Time\(us\):\s*([\d.]+)", re.M)
# "CPU Current Frequency List:\n    - 1800000\n ..." (printed because fix_freq=True)
# per-layer row: MacUsage "2.33/0.00/0.00" (blank for non-Conv), WorkLoad "100.0%/0.0%/0.0%" followed by RW(KB)
MAC_RE = re.compile(r"\d+\.\d+/\d+\.\d+/\d+\.\d+")
WORKLOAD_RE = re.compile(r"\d+\.\d%/\d+\.\d%/\d+\.\d%")
FREQ_RE = re.compile(r"^(CPU|NPU|DDR) Current Frequency List:\n((?:\s+- \d+\n?)+)", re.M)


@contextmanager
def connect_board(rknn_path: str, device_id: str, core_mask: str = "AUTO", **runtime_kwargs):
    """Load an RKNN model and initialize the runtime on the board.

    runtime_kwargs: perf_debug=True for the per-layer table, eval_mem=True for eval_memory().
    Keep them in separate sessions: both change the measured time.
    """
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime(target="rk3588", device_id=device_id, core_mask=CORE_MASK[core_mask], **runtime_kwargs)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    try:
        yield rknn
    finally:
        rknn.release()


def parse_frequencies(text: str) -> dict:
    return {name: [int(v) for v in re.findall(r"\d+", body)] for name, body in FREQ_RE.findall(text)}


def parse_summary_time_us(text: str) -> float:
    m = SUMMARY_TIME_RE.search(text)
    if m is None:
        raise ValueError("'Total Time(us):' not found in eval_perf output")
    return float(m.group(1))


def parse_layers(text: str) -> list[dict]:
    """Rows of the 'Network Layer Information Table' (perf_debug=True)."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[0].isdigit() or parts[3] not in ("CPU", "NPU", "GPU"):
            continue
        cycles = next((p for p in parts if re.fullmatch(r"\d+/\d+/\d+", p)), None)
        time_idx = parts.index(cycles) + 1 if cycles else None
        ddr, npu, total = (int(x) for x in cycles.split("/")) if cycles else (None, None, None)
        workload = next((p for p in parts if WORKLOAD_RE.fullmatch(p)), None)
        rows.append(
            {
                "id": int(parts[0]),
                "op": parts[1],
                "dtype": parts[2],
                "target": parts[3],
                "ddr_cycles": ddr,
                "npu_cycles": npu,
                "total_cycles": total,
                "time_us": int(parts[time_idx]) if time_idx else None,
                "mac_usage": next((p for p in parts if MAC_RE.fullmatch(p)), None),
                "workload": workload,
                "rw_kb": int(parts[parts.index(workload) + 1]) if workload else None,
                "name": parts[-1],
            }
        )
    return rows


def measure_latency(rknn: RKNN, repeats: int = PERF_REPEATS) -> dict:
    """Session 1 (perf_debug=False): NPU time of one frame, `repeats` eval_perf calls."""
    texts = [str(rknn.eval_perf(is_print=False, fix_freq=True)) for _ in range(repeats)]
    values = [parse_summary_time_us(t) for t in texts]
    median_us = statistics.median(values)
    return {
        "npu_latency_ms": round(median_us / 1000, 3),
        "npu_latency_ms_all": [round(v / 1000, 3) for v in values],
        "fps_latency": round(1e6 / median_us, 2),
        "frequencies_during_eval_perf": parse_frequencies(texts[-1]),
        "raw_last": texts[-1],
    }


def measure_layers(rknn: RKNN, repeats: int = PERF_REPEATS) -> tuple[dict, str]:
    """Session 2 (perf_debug=True): per-layer table from `repeats` eval_perf calls.

    time_us of a row is the median over the calls (time_us_all keeps every call). debug_total_us is debug-mode
    time, not the latency. Returns the summary and the raw text of the first call.
    """
    texts = [str(rknn.eval_perf(is_print=False, fix_freq=True)) for _ in range(repeats)]
    tables = [parse_layers(t) for t in texts]
    rows = tables[0]
    for t in tables[1:]:
        if [(r["id"], r["op"], r["name"]) for r in t] != [(r["id"], r["op"], r["name"]) for r in rows]:
            raise RuntimeError("layer table differs between eval_perf calls of one session")
    for i, row in enumerate(rows):
        row["time_us_all"] = [t[i]["time_us"] for t in tables]
        row["time_us"] = statistics.median(row["time_us_all"])
    totals = [float(m.group(1)) if (m := LAYERS_TIME_RE.search(t)) else None for t in texts]
    return {
        "debug_total_us": statistics.median(totals) if None not in totals else None,
        "debug_total_us_all": totals,
        "repeats": repeats,
        "n_layers": len(rows),
        "cpu_ops": sorted({r["op"] for r in rows if r["target"] == "CPU"} - {"InputOperator", "OutputOperator"}),
        "layers": rows,
    }, texts[0]


def measure_memory(rknn: RKNN) -> dict:
    """Session 3 (eval_mem=True): eval_memory result in MiB (RKNN API ref 2.10, p.19)."""
    mem = rknn.eval_memory(is_print=False)
    if not isinstance(mem, dict):
        raise RuntimeError(f"eval_memory returned {mem!r}; init_runtime needs eval_mem=True")
    return {k: round(v / 2**20, 3) for k, v in mem.items()}
