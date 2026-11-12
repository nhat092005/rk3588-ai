# rk3588-ai

PC side: dataset loaders, training, export, evaluation and NPU analysis for PPE detection on RK3588.

Planned layout (nothing migrated yet):

| Path | Content |
|---|---|
| `ai/core/` | Dataset loaders, trainer, evaluation metrics, RKNN library |
| `ai/models/` | One folder per model |
| `ai/research/` | Draft architectures |
| `ai/automation/` | Training and evaluation pipeline, configs |
| `ai/npu/` | NPU build, probe and analysis scripts |
| `experiments/`, `results/` | Measurements produced by this repo |
| `data/` | Datasets (only `DATA_CARD.md` tracked) |
| `scripts/`, `tests/` | Helper scripts and tests |

Shared pre/post-processing lives in `rk3588-common`, not here.
