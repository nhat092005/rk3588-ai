"""Single source of truth for rk3588-ai paths. Scripts import locations from here, never rebuild them."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
LEADERBOARDS_DIR = EXPERIMENTS_DIR / "leaderboards"
GOLDEN_DIR = EXPERIMENTS_DIR / "golden"
RESULTS_DIR = REPO_ROOT / "results"


def bench_dir(run) -> Path:
    """Pipeline outputs of a run (tier1/tier2/gpu files). run is a run_id or the run directory."""
    return RUNS_DIR / Path(run).name / "bench"
