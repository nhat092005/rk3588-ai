"""Repo states written into rk3588-ai result files."""

import sys

from ai.paths import REPO_ROOT
from rk3588_common.provenance import common_state, repo_arg, repo_state

# Code that runs. Generated output (data/, experiments/, results/) never
# makes the repo dirty.
CODE_PATHS = ("ai", "scripts")


def code_repos() -> dict:
    """Return {"ai": {...}, "common": {...}} for provenance()."""
    return {"ai": repo_state(REPO_ROOT, CODE_PATHS), "common": common_state()}


def board_repo_args() -> list[str]:
    """Return --repo arguments for a board script."""
    return [
        arg
        for name, state in code_repos().items()
        for arg in ("--repo", repo_arg(name, state))
    ]


if (
    __name__ == "__main__"
):  # used by the Makefile: python -m ai.core.provenance board-args
    if sys.argv[1:] != ["board-args"]:
        raise SystemExit("usage: python -m ai.core.provenance board-args")
    print(" ".join(board_repo_args()))
