"""Run a command, show its output live, and save a cleaned log file.

Usage: python scripts/logtee.py <log path, no timestamp> -- <cmd> [...]
Writes <path>_<timestamp>.log (never overwrites older logs) with a
command/git header; ANSI codes stripped and carriage-return progress
updates collapsed to final state. Exit code = command's.
"""

import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.core.provenance import code_repos

ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")


def clean_line(line: bytes) -> bytes:
    r"""Keep terminal final line text: bytes after the last '\r'."""
    line = ANSI_RE.sub(b"", line)
    body = line.rstrip(b"\r")
    return body.rsplit(b"\r", 1)[-1]


def main() -> int:
    """Execute command, tee output to stdout and cleaned log file."""
    sep = sys.argv.index("--")
    base, cmd = Path(sys.argv[1]), sys.argv[sep + 1 :]
    stamp = datetime.datetime.now()
    log_path = base.parent / f"{base.name}_{stamp:%Y%m%d-%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("wb") as log:
        repos = ", ".join(
            f"{name}={state['commit']} dirty={state['dirty']}"
            for name, state in code_repos().items()
        )
        started = stamp.isoformat(timespec="seconds")
        rknn_log = os.environ.get("RKNN_LOG_LEVEL", "")
        header = (
            f"# command: {' '.join(cmd)}\n# started: {started}\n"
            f"# repos: {repos}\n"
            f"# env: RKNN_LOG_LEVEL={rknn_log}\n\n"
        )
        log.write(header.encode())
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        pending = b""
        while chunk := os.read(proc.stdout.fileno(), 65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for line in lines:
                log.write(clean_line(line) + b"\n")
            log.flush()
        if pending:
            log.write(clean_line(pending) + b"\n")
        code = proc.wait()
        finished = datetime.datetime.now().isoformat(timespec="seconds")
        log.write(f"\n# exit_code: {code}\n# finished: {finished}\n".encode())
    print(f"[log] {log_path}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
