"""CLI: check the board is ready before NPU steps (make board-check [LOCK=1]).

Checks adb connectivity (needed by tier-1 RKNN-Toolkit2 calls from the PC) and rknn_server
running on the board, plus, with --require-lock, CPU/NPU/DDR governors locked to
performance/userspace (needed for tier-2 measurements). Exits 1 with the fix command on failure.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import rknn.api

ADB = Path(rknn.api.__file__).parents[1] / "3rdparty" / "platform-tools" / "adb" / "linux-x86_64" / "adb"
GOVERNORS = (
    "/sys/devices/system/cpu/cpufreq/policy*/scaling_governor "
    "/sys/class/devfreq/fdab0000.npu/governor /sys/class/devfreq/dmc/governor"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("board", help="ssh target, e.g. minhnhat@100.67.251.37")
    parser.add_argument("device_id", help="adb address, e.g. 100.67.251.37:5555")
    parser.add_argument("--require-lock", action="store_true")
    args = parser.parse_args()
    fixes = []

    subprocess.run([str(ADB), "connect", args.device_id], capture_output=True, text=True)
    devices = subprocess.run([str(ADB), "devices"], capture_output=True, text=True).stdout
    if f"{args.device_id}\tdevice" not in devices:
        fixes.append("adb cannot reach the board: sudo bash deploy/sh/start_rknn_server.sh")

    remote = subprocess.run(
        ["ssh", args.board, f"pgrep -x rknn_server >/dev/null && echo SERVER_OK; cat {GOVERNORS}"],
        capture_output=True,
        text=True,
    )
    if "SERVER_OK" not in remote.stdout:
        fixes.append("rknn_server is not running: sudo bash deploy/sh/start_rknn_server.sh")
    governors = [g for g in remote.stdout.split() if g != "SERVER_OK"]
    # tier 1 (eval_perf fix_freq=True) leaves NPU/DDR on 'userspace' at their maximum frequency: also fixed
    locked = bool(governors) and all(g in ("performance", "userspace") for g in governors)
    if args.require_lock and not locked:
        fixes.append(
            f"frequencies not locked (governors: {' '.join(governors)}): sudo -n /usr/local/sbin/lock_freq.sh lock"
        )

    print(
        f"adb: {'ok' if not any('adb' in f for f in fixes) else 'FAIL'}, rknn_server: "
        f"{'ok' if 'SERVER_OK' in remote.stdout else 'FAIL'}, frequencies locked: {locked}"
    )
    if fixes:
        print(
            "Run on the board (in the board directory, BOARD_DIR of board.mk):\n  " + "\n  ".join(dict.fromkeys(fixes))
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
