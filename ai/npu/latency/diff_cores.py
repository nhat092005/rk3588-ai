"""List the registers that differ between the 3 cores of a task.

For each task of each 3-core submit (mask 0x7), compares the command
words of cores 0, 1, 2 position by position and prints the words that
differ as (target, register, value core 0, core 1, core 2). Rows and
addresses that differ mean the task is split; a task whose only
difference is the PC register 0x10 (next-task address) is not.

Compares words up to the shortest of the three and adds a "length
differs" line when the three tasks have different word counts. Tasks are
paired by position, so a submit whose cores have unequal task counts
raises ValueError.
"""

import argparse
from pathlib import Path

from ai.npu.latency.trace_log import CORE_SLOTS, parse_trace, split_word


def diff_lines(path: Path) -> list[str]:
    """Return the report lines for the log at path."""
    lines: list[str] = []
    for s in parse_trace(path):
        if s.mask != "0x7":
            continue
        slots = [s.tasks[k] for k in CORE_SLOTS]
        if len({len(t) for t in slots}) != 1:
            raise ValueError(f"submit {s.n}: cores have unequal task counts")
        for t0, t1, t2 in zip(*slots, strict=True):
            diffs = []
            for w0, w1, w2 in zip(t0.cmd, t1.cmd, t2.cmd, strict=False):
                if w0 != w1 or w0 != w2:
                    tg, v0, reg = split_word(w0)
                    diffs.append(
                        (
                            hex(tg),
                            hex(reg),
                            v0,
                            split_word(w1)[1],
                            split_word(w2)[1],
                        )
                    )
            lines.append(
                f"submit {s.n} task {t0.idx} op {t0.op} en {t0.en}"
                f" ndiff {len(diffs)}"
            )
            lines += [f"    {d}" for d in diffs]
            if len({len(t0.cmd), len(t1.cmd), len(t2.cmd)}) != 1:
                lens = (len(t0.cmd), len(t1.cmd), len(t2.cmd))
                lines.append(f"    length differs: {lens}")
    return lines


def main() -> None:
    """Print the report of the log given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    print("\n".join(diff_lines(parser.parse_args().log)))


if __name__ == "__main__":
    main()
