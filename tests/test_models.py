"""Every config names a model folder with WEIGHTS (models/README.md)."""

import importlib
from pathlib import Path

import pytest
import yaml

from ai.paths import REPO_ROOT

CONFIGS = sorted((REPO_ROOT / "ai" / "automation" / "configs").glob("*.yaml"))


def test_configs_exist():
    assert {p.stem for p in CONFIGS} >= {
        "ppe5_yolov8n_baseline",
        "ppe5_yolov8n_baseline_200ep",
    }


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_config_model_is_importable(config):
    cfg = yaml.safe_load(config.read_text())
    assert {"dataset", "model", "epochs", "seed"} <= cfg.keys()
    module = importlib.import_module(
        f"ai.models.{cfg['model']}.{cfg['model']}"
    )
    assert isinstance(module.WEIGHTS, str)
    assert Path(module.__file__).parent.name == cfg["model"]


def test_baseline_weights():
    from ai.models.yolov8n_baseline import yolov8n_baseline

    assert yolov8n_baseline.WEIGHTS == "yolov8n.pt"
