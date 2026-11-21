"""Show how the 3 cores share each convolution task of a 3-core submit.

For the first inference of a log from npu_trace.c, groups the conv tasks
of the 3-core submits (mask 0x7) by kernel size, input width and what
each core receives: input rows, output rows, top pad, the first input
row (from the feature address) and whether the weight address differs.

Reading the numbers: input rows larger than output rows mean the core
reads halo rows of its neighbours; a feature address that differs means
an H split, a weight address that differs with one shared feature
address means an output-channel split.

The first input row is the address difference divided by the row size
in bytes, taken as input width * 16 (int8 data in groups of 16
channels); it is None when the difference is not a multiple of that. The
guess does not hold for the 3-channel network input (width 640), whose
first_row values are not to be read.
"""

import argparse
from collections import Counter
from pathlib import Path

from ai.npu.latency.summarize import inference_length
from ai.npu.latency.trace_log import CORE_SLOTS, Task, parse_trace, read_reg

CNA, CORE = 0x201, 0x801  # register blocks, TRM Part1 ch. RKNN
CNA_DATA_SIZE0 = 0x1020  # datain_width 26:16, datain_height 10:0
CNA_WEIGHT_SIZE2 = 0x1038  # kernel width 28:24, kernel height 20:16
CNA_PAD_CON0 = 0x1068  # pad_left 7:4, pad_top 3:0
CNA_FEATURE_ADDR = 0x1070
CNA_DCOMP_ADDR0 = 0x1110  # weight base address
CORE_DATAOUT_SIZE0 = 0x3014  # dataout_height 31:16, dataout_width 15:0
CONV_ENABLE = ("0x1d", "0xd")
ROW_BYTES_PER_WIDTH = 16


def _reg(task: Task, target: int, reg: int) -> int:
    """Return a register of the task, or raise if it is not written."""
    value = read_reg(task.cmd, target, reg)
    if value is None:
        raise ValueError(f"task {task.idx}: register {target:#x}/{reg:#x}")
    return value


def conv_rows(trio: tuple[Task, ...]) -> dict:
    """Return the per-core numbers of one conv task on 3 cores."""
    size2 = _reg(trio[0], CNA, CNA_WEIGHT_SIZE2)
    data0 = [_reg(t, CNA, CNA_DATA_SIZE0) for t in trio]
    width = (data0[0] >> 16) & 0x7FF
    feature = [_reg(t, CNA, CNA_FEATURE_ADDR) for t in trio]
    row_bytes = width * ROW_BYTES_PER_WIDTH
    first_rows = []
    for addr in feature:
        diff = addr - feature[0]
        first_rows.append(diff // row_bytes if diff % row_bytes == 0 else None)
    out = [_reg(t, CORE, CORE_DATAOUT_SIZE0) for t in trio]
    weights = {_reg(t, CNA, CNA_DCOMP_ADDR0) for t in trio}
    return {
        "kernel": f"{(size2 >> 16) & 0x1F}x{(size2 >> 24) & 0x1F}",
        "in_w": width,
        "in_h": tuple(v & 0x7FF for v in data0),
        "out_h": tuple(((v >> 16) & 0xFFFF) + 1 for v in out),
        "pad_top": tuple(_reg(t, CNA, CNA_PAD_CON0) & 0xF for t in trio),
        "first_row": tuple(first_rows),
        "weights": len(weights),
    }


def split_lines(path: Path) -> list[str]:
    """Return the report lines for the log at path."""
    submits = parse_trace(path)
    groups: Counter = Counter()
    for s in submits[: inference_length(submits)]:
        if s.mask != "0x7":
            continue
        slots = [s.tasks[k] for k in CORE_SLOTS]
        if len({len(t) for t in slots}) != 1:
            raise ValueError(f"submit {s.n}: cores have unequal task counts")
        for trio in zip(*slots, strict=True):
            if trio[0].en in CONV_ENABLE:
                groups[tuple(conv_rows(trio).items())] += 1
    lines = [str(path)]
    for key, count in sorted(groups.items(), key=lambda g: g[0]):
        fields = " ".join(f"{name}={value}" for name, value in key)
        lines.append(f"  {count:3d} x {fields}")
    return lines


def main() -> None:
    """Print the report of each log given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    for log in parser.parse_args().logs:
        print("\n".join(split_lines(log)))


if __name__ == "__main__":
    main()
