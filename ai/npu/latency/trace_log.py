"""Reader of the log written by ai/board/probe/npu_trace.c.

The log has one SUBMIT line per submit ioctl. After it come, for each
task, a TASK line and a CMD line with the register command words of the
task. A log made before the 400-word cut was removed from npu_trace.c
(for example trace_c012.log) holds at most 400 words per task, so a
longer task is compared only up to the cut. A "CMDCUT" line marks a task
whose words run past its buffer. A "CMD unmapped" or "TASKOBJ unmapped"
line raises ValueError.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

# Slot of cores 0, 1, 2 in subcore_task[] of a 3-core job:
# subcore_task[core_index + 2] in the driver, rknpu_job.c:86, 322-324.
CORE_SLOTS = (2, 3, 4)
# Register command word: target (16 bits), value (32 bits), register.
TARGET_DPU = 0x1001
DPU_DATA_CUBE_HEIGHT = 0x4034  # TRM Part1, RKNN_dpu_data_cube_height
DPU_HEIGHT_MASK = 0x1FFF  # the height is bits 12:0 of the value

_SUBMIT = re.compile(r"n=(\d+) core_mask=(0x[0-9a-f]+)")
_TASK = re.compile(
    r"sub=(\d) idx=(\d+) op=(\d+) .* en=(0x[0-9a-f]+) .* amount=(\d+)"
)


class Task(NamedTuple):
    """One task of a SUBMIT with its register command words."""

    idx: int
    op: int
    en: str
    amount: int
    cmd: list[int]


@dataclass
class Submit:
    """One submit ioctl: core mask and its tasks by subcore slot."""

    n: int
    mask: str
    tasks: dict[int, list[Task]] = field(
        default_factory=lambda: defaultdict(list)
    )


def parse_trace(path: Path) -> list[Submit]:
    """Read every SUBMIT of the log with the tasks of each slot."""
    lines = path.read_text().splitlines()
    submits: list[Submit] = []
    for i, line in enumerate(lines):
        if "unmapped" in line:
            raise ValueError(f"{path}:{i + 1}: {line!r}, rerun the trace")
        if line.startswith("SUBMIT"):
            m = _SUBMIT.search(line)
            submits.append(Submit(int(m[1]), m[2]))
        elif line.startswith("TASK"):
            m = _TASK.search(line)
            cmd = [int(w, 16) for w in lines[i + 1].split()[1:]]
            submits[-1].tasks[int(m[1])].append(
                Task(int(m[2]), int(m[3]), m[4], int(m[5]), cmd)
            )
    return submits


def split_word(word: int) -> tuple[int, int, int]:
    """Split a command word into (target, value, register)."""
    return word >> 48, (word >> 16) & 0xFFFFFFFF, word & 0xFFFF


def read_reg(cmd: list[int], target: int, reg: int) -> int | None:
    """Return the value of the first write to (target, reg), if any."""
    for word in cmd:
        tg, value, r = split_word(word)
        if tg == target and r == reg:
            return value
    return None
