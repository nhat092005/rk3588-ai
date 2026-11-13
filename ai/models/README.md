# models/

Model architecture definitions without training logic. One folder per model, named after the model, holding `<name>.py` (`WEIGHTS`, optional `PRETRAINED`) and, when the model has its own architecture, `<name>.yaml`. A model that uses a stock Ultralytics checkpoint has only the `.py`.

Every model here is trained with an official config in `ai/automation/configs/` whose `model:` field is the folder name (checked by `tests/test_models.py`). The pipeline imports `ai.models.<name>.<name>` and passes its `WEIGHTS` to the trainer. A new model needs the folder, an empty `__init__.py`, the `.py`, and the yaml if any.

| Folder | Description | Status |
|---|---|---|
| `yolov8n_baseline/` | Baseline YOLOv8 nano (SiLU), `yolov8n.pt` | Reference of the gate G4 |

The other legacy models (ReLU, LeakyReLU, ReLU6, no-DFL, all-core, scratch controls, YOLOv8s, YOLO26n) stay in `legacy-final:ai/models/` with their statuses until their research line migrates. Weights download through Ultralytics into `~/.cache/ultralytics/weights/`.
