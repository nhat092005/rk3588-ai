"""Count the tasks of a 3-core trace and the rows each core gets.

For the first inference of a log from npu_trace.c, counts tasks by
(core mask, kind) and, for 3-core submits, the rows of each core from
the DPU data cube height. The kind comes from the task enable mask:
0x18 writes DPU only (elementwise), 0x60 PPU only (pool), 0x1d and 0xd
CNA, CORE and DPU (conv).

The log holds the same inference several times: the submits of the first
one are found as the shortest repeating block of the submit sequence
(same mask, task counts and enable masks), and only they are read. A log
that does not repeat is read whole.
"""

import argparse
from collections import Counter
from pathlib import Path

from ai.npu.latency.trace_log import (
    CORE_SLOTS,
    DPU_DATA_CUBE_HEIGHT,
    DPU_HEIGHT_MASK,
    TARGET_DPU,
    Submit,
    parse_trace,
    read_reg,
)

KIND = {
    "0x18": "elementwise",
    "0x60": "pool",
    "0x1d": "conv",
    "0xd": "conv_nordma",
}


# First slot of subcore_task[] that holds the tasks of a core mask,
# rknpu_job.c:86 and 312-327: slot core_index for 1 or 2 cores, and
# core_index + 2 for 3 cores.
MASK_SLOT = {"0x1": 0, "0x2": 1, "0x4": 2, "0x3": 0, "0x7": 2}


def _signature(s: Submit) -> tuple:
    """Return what makes two submits the same step of an inference."""
    return (
        s.mask,
        tuple(
            (slot, tuple((t.en, t.amount) for t in tasks))
            for slot, tasks in sorted(s.tasks.items())
        ),
    )


def inference_length(submits: list[Submit]) -> int:
    """Return the number of submits of one inference of the log."""
    sig = [_signature(s) for s in submits]
    for period in range(1, len(sig) + 1):
        if len(sig) % period == 0 and all(
            sig[i] == sig[i % period] for i in range(len(sig))
        ):
            return period
    return len(sig)


def summary_lines(path: Path) -> list[str]:
    """Return the report lines for the log at path."""
    submits = parse_trace(path)
    first = inference_length(submits)
    by_kind: Counter = Counter()
    triplets: Counter = Counter()
    for s in submits[:first]:
        if s.mask not in MASK_SLOT:
            raise ValueError(f"submit {s.n}: core mask {s.mask} not handled")
        for task in s.tasks.get(MASK_SLOT[s.mask], []):
            by_kind[(s.mask, KIND.get(task.en, task.en))] += 1
        if s.mask != "0x7":
            continue
        slots = [s.tasks[k] for k in CORE_SLOTS]
        if len({len(t) for t in slots}) != 1:
            raise ValueError(f"submit {s.n}: cores have unequal task counts")
        for trio in zip(*slots, strict=True):
            rows = [
                read_reg(t.cmd, TARGET_DPU, DPU_DATA_CUBE_HEIGHT) for t in trio
            ]
            if None in rows:
                continue
            triplets[tuple((r & DPU_HEIGHT_MASK) + 1 for r in rows)] += 1
    return [
        str(path),
        "  submits per inference:"
        f" {first}  tasks by (mask, kind) for first inference:"
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
