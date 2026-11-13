"""CLI: Model Complexity of a run (Table E).

Params/GFLOPs of the fused model at 640x640 (MACs = GFLOPs/2); sizes of
FP32 .onnx and .rknn, not .pt (Ultralytics saves best.pt in FP16,
torch_utils.py strip_optimizer). Writes runs/<run_id>/complexity.json.
Run after export_npu.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from ai.core.provenance import code_repos
from ai.paths import RUNS_DIR
from rk3588_common.model_io import IMGSZ, PROTOCOL_VERSION


def size_mb(path: Path) -> float | None:
    """Return file size in MiB rounded to 3 decimals, or None."""
    return round(path.stat().st_size / 2**20, 3) if path.exists() else None


def main() -> None:
    """Compute model params, GFLOPs, MACs, and file sizes for a run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    weights = run_dir / "weights"
    model = YOLO(weights / "best.pt").model.float().fuse()
    gflops = get_flops(model, IMGSZ)
    if gflops == 0:  # get_flops returns 0.0 silently when thop fails
        raise RuntimeError(
            "get_flops returned 0; check that ultralytics-thop is installed"
        )

    rknn = {
        p.name: size_mb(next(p.glob("*.rknn")))
        for p in sorted(weights.glob("rknn_*"))
        if any(p.glob("*.rknn"))
    }
    record = {
        "run_id": args.run_id,
        "protocol_version": PROTOCOL_VERSION,
        "repos": code_repos(),
        "imgsz": IMGSZ,
        "fused": True,
        "params_m": round(get_num_params(model) / 1e6, 3),
        "gflops": round(gflops, 3),
        "macs_g": round(gflops / 2, 3),
        "onnx_fp32_mb": size_mb(weights / "best.onnx"),
        "rknn_mb": rknn,
    }
    out = run_dir / "complexity.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
