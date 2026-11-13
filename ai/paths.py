"""Single source of truth for rk3588-ai paths.

Scripts import locations from here and never rebuild them.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
LEADERBOARDS_DIR = EXPERIMENTS_DIR / "leaderboards"
GOLDEN_DIR = EXPERIMENTS_DIR / "golden"
RESULTS_DIR = REPO_ROOT / "results"


def bench_dir(run: str | Path) -> Path:
    """Return bench directory of a run (given run_id or run path)."""
    return RUNS_DIR / Path(run).name / "bench"
