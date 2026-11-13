"""Resolve dataset.yaml paths produced by scripts/data/prepare_<name>.py."""

from pathlib import Path

from ai.paths import DATA_DIR


def dataset_yaml_path(name: str) -> Path:
    path = DATA_DIR / name / "processed" / "dataset.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run scripts/data/prepare_{name}.py first.")
    return path
