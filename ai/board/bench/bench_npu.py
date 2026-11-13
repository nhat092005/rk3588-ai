"""Tier-2 latency of a .rknn model, run ON THE BOARD with RKNNLite (Tables G, H, L).

20 warmup frames, then one timed pass over the test set (pre/npu/post/e2e per image, file
read/decode excluded). Resource usage (VmRSS, CPU, NPU temperature) is sampled during the pass.
Usage (on the board, in <board dir>/ai): ../.venv/bin/python -m ai.board.bench.bench_npu --model <dir> --images <dir> --out <json>
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import cv2
import yaml
from rknnlite.api import RKNNLite

from ai.board.bench import WARMUP
from rk3588_common.board_env import ResourceMonitor, frequency_state, summarize, system_info
from rk3588_common.model_io import boxes_normalized, list_images, postprocess, preprocess
from rk3588_common.provenance import add_provenance_args, board_provenance

CORE_MASK = {
    "AUTO": RKNNLite.NPU_CORE_AUTO,
    "CORE_0": RKNNLite.NPU_CORE_0,
    "CORE_1": RKNNLite.NPU_CORE_1,
    "CORE_2": RKNNLite.NPU_CORE_2,
    "CORE_0_1": RKNNLite.NPU_CORE_0_1,
    "CORE_0_1_2": RKNNLite.NPU_CORE_0_1_2,
}


def load_model(model_dir: Path, core_mask: str) -> tuple[RKNNLite, bool]:
    """Returns the runtime and whether boxes are normalized (Ultralytics INT8 export)."""
    meta = yaml.safe_load((model_dir / "metadata.yaml").read_text())
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(next(model_dir.glob("*.rknn")))) != 0:
        raise RuntimeError("load_rknn failed")
    if rknn.init_runtime(core_mask=CORE_MASK[core_mask]) != 0:
        raise RuntimeError("init_runtime failed")
    return rknn, boxes_normalized(meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="rknn_<precision>/ folder")
    parser.add_argument("--images", required=True)
    parser.add_argument("--core-mask", default="AUTO", choices=list(CORE_MASK))
    parser.add_argument("--out", required=True, type=Path)
    add_provenance_args(parser)
    args = parser.parse_args()

    images = list_images(args.images)
    rknn, normalized = load_model(args.model, args.core_mask)
    freq_before = frequency_state()

    warm = cv2.imread(str(images[0]))
    for _ in range(WARMUP):
        inp, meta = preprocess(warm)
        postprocess(rknn.inference(inputs=[inp])[0], meta, normalized)

    pre, npu, post, e2e, n_det = [], [], [], [], []
    with ResourceMonitor() as monitor:
        for path in images:
            bgr = cv2.imread(str(path))
            t0 = time.perf_counter()
            inp, meta = preprocess(bgr)
            t1 = time.perf_counter()
            out = rknn.inference(inputs=[inp])[0]
            t2 = time.perf_counter()
            det = postprocess(out, meta, normalized)
            t3 = time.perf_counter()
            pre.append((t1 - t0) * 1e3)
            npu.append((t2 - t1) * 1e3)
            post.append((t3 - t2) * 1e3)
            e2e.append((t3 - t0) * 1e3)
            n_det.append(len(det))
    sdk_version = rknn.get_sdk_version()
    rknn.release()

    e2e_stats = summarize(e2e)
    record = {
        **board_provenance(args, next(args.model.glob("*.rknn"))),
        "runtime": "rknnlite",
        "model": str(args.model),
        "normalized_boxes": normalized,
        "core_mask": args.core_mask,
        "contexts": 1,
        "images": args.images,
        "n_images": len(images),
        "warmup": WARMUP,
        "measured_at": datetime.datetime.now().isoformat(),
        "sdk_version": str(sdk_version),
        "system": system_info(),
        "frequency_before": freq_before,
        "frequency_after": frequency_state(),
        "latency_ms": {"pre": summarize(pre), "npu": summarize(npu), "post": summarize(post), "e2e": e2e_stats},
        "fps_latency": round(1000 / e2e_stats["mean"], 2),
        "detections_per_image_mean": round(sum(n_det) / len(n_det), 2),
        "resource": monitor.result(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Done: {args.out}  e2e mean {e2e_stats['mean']} ms, p95 {e2e_stats['p95']} ms")


if __name__ == "__main__":
    main()
