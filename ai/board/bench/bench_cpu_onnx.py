"""Tier-2 latency of FP32 .onnx on the board CPU with ONNX Runtime.

Same pre/post-processing, warmup, and test pass as bench_npu.py; only
inference differs (Table H). Usage (on the board, in <board dir>/ai):
    ../.venv/bin/python -m ai.board.bench.bench_cpu_onnx \
        --model best.onnx --images <dir> --threads 4 --out <json>
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ai.board.bench import WARMUP
from rk3588_common.board_env import (
    ResourceMonitor,
    frequency_state,
    summarize,
    system_info,
)
from rk3588_common.model_io import list_images, postprocess, preprocess
from rk3588_common.provenance import add_provenance_args, board_provenance


def main() -> None:
    """Run ONNX Runtime CPU latency benchmark and write JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, type=Path, help="FP32 best.onnx"
    )
    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--threads", type=int, default=4, help="ONNX Runtime intra-op threads"
    )
    parser.add_argument("--out", required=True, type=Path)
    add_provenance_args(parser)
    args = parser.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.model), opts, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    def infer(nhwc: np.ndarray) -> np.ndarray:
        nchw = (
            np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2), dtype=np.float32)
            / 255
        )  # part of pre
        return session.run(None, {input_name: nchw})[0]

    images = list_images(args.images)
    freq_before = frequency_state()
    warm = cv2.imread(str(images[0]))
    for _ in range(WARMUP):
        inp, meta = preprocess(warm)
        postprocess(infer(inp), meta, normalized=False)

    pre, cpu, post, e2e = [], [], [], []
    with ResourceMonitor() as monitor:
        for path in images:
            bgr = cv2.imread(str(path))
            t0 = time.perf_counter()
            inp, meta = preprocess(bgr)
            t1 = time.perf_counter()
            out = infer(inp)
            t2 = time.perf_counter()
            postprocess(out, meta, normalized=False)
            t3 = time.perf_counter()
            pre.append((t1 - t0) * 1e3)
            cpu.append((t2 - t1) * 1e3)
            post.append((t3 - t2) * 1e3)
            e2e.append((t3 - t0) * 1e3)

    e2e_stats = summarize(e2e)
    record = {
        **board_provenance(args, args.model),
        "runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider",
        "model": str(args.model),
        "intra_op_threads": args.threads,
        "images": args.images,
        "n_images": len(images),
        "warmup": WARMUP,
        "measured_at": datetime.datetime.now().isoformat(),
        "system": system_info(),
        "frequency_before": freq_before,
        "frequency_after": frequency_state(),
        "latency_ms": {
            "pre": summarize(pre),
            "inference": summarize(cpu),
            "post": summarize(post),
            "e2e": e2e_stats,
        },
        "fps_latency": round(1000 / e2e_stats["mean"], 2),
        "resource": monitor.result(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"Done: {args.out}  e2e mean {e2e_stats['mean']} ms, "
        f"p95 {e2e_stats['p95']} ms"
    )


if __name__ == "__main__":
    main()
