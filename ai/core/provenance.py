"""Repo states written into rk3588-ai result files (rk3588_common.provenance schema)."""

from ai.paths import REPO_ROOT
from rk3588_common.provenance import common_state, repo_state

# Code that runs. Generated output (data/, experiments/, results/) never makes the repo dirty.
CODE_PATHS = ("ai", "scripts")


def code_repos() -> dict:
    """{"ai": {...}, "common": {...}} for provenance(): the repos whose code an ai process runs."""
    return {"ai": repo_state(REPO_ROOT, CODE_PATHS), "common": common_state()}
