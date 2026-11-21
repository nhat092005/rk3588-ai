"""Count the tasks of a 3-core trace and the rows each core gets.

For the first inference of a log from npu_trace.c, counts tasks by
(core mask, kind) and, for 3-core submits, the rows of each core from
the DPU data cube height. The kind comes from the task enable mask:
0x18 writes DPU only (elementwise), 0x60 PPU only (pool), 0x1d and 0xd
CNA, CORE and DPU (conv).

Assumes the log holds two identical inferences: only the first half of
the submits is read. Assumes masks 0x1 and 0x7 (task slots 0 and 2).
"""

import argparse
from collections import Counter
from pathlib import Path

from ai.npu.latency.trace_log import (
    CORE_SLOTS,
    DPU_DATA_CUBE_HEIGHT,
    TARGET_DPU,
    parse_trace,
    read_reg,
)

KIND = {
    "0x18": "elementwise",
    "0x60": "pool",
    "0x1d": "conv",
    "0xd": "conv_nordma",
}


def summary_lines(path: Path) -> list[str]:
    """Return the report lines for the log at path."""
    submits = parse_trace(path)
    half = (max(s.n for s in submits) + 1) // 2
    by_kind: Counter = Counter()
    triplets: Counter = Counter()
    for s in submits:
        if s.n >= half:
            continue
        slot = 0 if s.mask == "0x1" else 2
        for task in s.tasks.get(slot, []):
            by_kind[(s.mask, KIND.get(task.en, task.en))] += 1
        if s.mask != "0x7":
            continue
        for trio in zip(*[s.tasks[k] for k in CORE_SLOTS], strict=False):
            rows = [
                read_reg(t.cmd, TARGET_DPU, DPU_DATA_CUBE_HEIGHT) for t in trio
            ]
            if None in rows:
                continue
            triplets[tuple(r + 1 for r in rows)] += 1
    return [
        str(path),
        "  submits per inference:"
        f" {half}  tasks by (mask, kind) for first inference:"
        f" {dict(by_kind)}",
        f"  height triplets (rows per core, count): {dict(triplets)}",
    ]


def main() -> None:
    """Print the report of each log given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    for log in parser.parse_args().logs:
        print("\n".join(summary_lines(log)))


if __name__ == "__main__":
    main()
