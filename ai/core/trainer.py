"""Central model training and evaluation interface using Ultralytics."""

from dataclasses import dataclass

from ultralytics import YOLO

from .dataset import dataset_yaml_path


@dataclass
class TrainConfig:
    """Configuration parameters for an Ultralytics YOLO training run."""

    # Model checkpoint name ("yolov8n.pt") or architecture YAML path.
    model: str
    dataset: str  # Dataset identifier, e.g. "ppe5"
    epochs: int
    imgsz: int = 640
    batch: int = 16
    seed: int = 0  # Ultralytics default
    device: str = "0"  # Target device, e.g. "0" for GPU, "cpu" for CPU
    # Ultralytics default; auto picks the optimizer and ignores lr0 and
    # momentum, so they are not set here
    optimizer: str = "auto"
    project: str | None = None  # Output directory for run artifacts
    run_name: str = "train"
    # weights to load into a yaml-built model; None keeps default
    pretrained: str | None = None


def train(cfg: TrainConfig) -> YOLO:
    """Train a YOLO model according to cfg and return the model."""
    data_yaml = dataset_yaml_path(cfg.dataset)
    model = YOLO(cfg.model)
    extra = {"pretrained": cfg.pretrained} if cfg.pretrained else {}
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
        device=cfg.device,
        optimizer=cfg.optimizer,
        project=cfg.project,
        name=cfg.run_name,
        exist_ok=True,
        **extra,
    )
    return model


def evaluate(
    model: YOLO,
    dataset: str,
    split: str = "test",
    project: str | None = None,
    run_name: str = "test_eval",
) -> object:
    """Validate a trained YOLO model on a dataset split."""
    data_yaml = dataset_yaml_path(dataset)
    return model.val(
        data=str(data_yaml),
        split=split,
        project=project,
        name=run_name,
        exist_ok=True,
    )
