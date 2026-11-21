"""Write the submit-trace fixtures from a legacy-final checkout.

Run once from this repo's root:
    python3 tests/make_trace_fixtures.py <legacy>
<legacy> is the legacy checkout (HEAD at legacy-final). The fixture log
is a slice of its trace_c012.log: the SUBMITS below, written twice
(renumbered), because the legacy summarize.py reads the first half of
the log as one inference. The expected outputs come from running the
legacy summarize.py and diff_cores.py on that slice.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import trace_cases as cases

TRACE = "experiments/npu/latency/submit_trace"
# One single-core and three 3-core submits (elementwise and pool, and
# 1x1 conv with pool) of the first inference of trace_c012.log.
SUBMITS = (1, 3, 8, 46)


def check_legacy(legacy: Path) -> None:
    """Fail unless the checkout is the frozen legacy-final tag."""
    tag = subprocess.check_output(
        ["git", "-C", str(legacy), "describe", "--exact-match", "--tags"],
        text=True,
    )
    assert tag.strip() == "legacy-final", (
        f"legacy HEAD is at {tag.strip()}, not legacy-final"
    )


def submit_blocks(log: Path) -> dict[int, list[str]]:
    """Split a trace log into its SUBMIT blocks, keyed by n."""
    blocks: dict[int, list[str]] = {}
    cur: list[str] | None = None
    for line in log.read_text().splitlines():
        if line.startswith("SUBMIT"):
            cur = blocks.setdefault(int(re.search(r"n=(\d+)", line)[1]), [])
        elif line.startswith("STATS"):
            cur = None
        if cur is not None:
            cur.append(line)
    return blocks


def renumber(block: list[str], n: int) -> list[str]:
    """Return the block with its SUBMIT header numbered n."""
    return [re.sub(r"n=\d+", f"n={n}", block[0], count=1), *block[1:]]


def main() -> None:
    """Write the fixture log and the outputs of the legacy scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    legacy = parser.parse_args().legacy
    check_legacy(legacy)
    blocks = submit_blocks(legacy / TRACE / "trace_c012.log")
    lines: list[str] = []
    for n in range(2 * len(SUBMITS)):
        lines += renumber(blocks[SUBMITS[n % len(SUBMITS)]], n)
    cases.TRACE_LOG.write_text("\n".join(lines) + "\n")
    for script, out in (
        ("summarize.py", cases.TRACE_SUMMARY),
        ("diff_cores.py", cases.TRACE_DIFF),
    ):
        text = subprocess.check_output(
            [
                sys.executable,
                str(legacy / TRACE / script),
                str(cases.TRACE_LOG),
            ],
            text=True,
        )
        out.write_text(text.replace(str(cases.TRACE_LOG), "<log>"))


if __name__ == "__main__":
    main()
