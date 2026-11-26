"""Resolve and read the data.yaml of a dataset folder under data/."""

from pathlib import Path

import yaml

from ai.paths import DATA_DIR


def dataset_yaml_path(name: str) -> Path:
    """Return path to data/<name>/data.yaml, raising if missing."""
    path = DATA_DIR / name / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Put the dataset folder in data/."
        )
    return path


def load_dataset_yaml(name: str) -> dict:
    """Load data.yaml with "path" set to the absolute dataset root.

    The files have no "path" key (no absolute path in the repo), so the
    root is the yaml's directory, like Ultralytics check_det_dataset.
    """
    path = dataset_yaml_path(name)
    data = yaml.safe_load(path.read_text())
    data["path"] = str((path.parent / data.get("path", ".")).resolve())
    return data
