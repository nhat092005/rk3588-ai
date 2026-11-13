"""Throughput FPS with several RKNNLite contexts in parallel, run ON THE BOARD (Table I).

One thread per context, each running pre->NPU->post on cached decoded frames (file decode excluded).
Throughput FPS = frames / wall time. Usage: python -m ai.board.bench.bench_throughput --model <dir> --images <dir> --core-masks <masks> --out <json>
"""

import argparse
import datetime
import itertools
import json
import threading
import time
from pathlib import Path

import cv2

from ai.board.bench import WARMUP
from ai.board.bench.bench_npu import CORE_MASK, load_model
from rk3588_common.board_env import ResourceMonitor, frequency_state, summarize, system_info
from rk3588_common.model_io import list_images, postprocess, preprocess
from rk3588_common.provenance import add_provenance_args, board_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="rknn_<precision>/ folder")
    parser.add_argument("--images", required=True)
    parser.add_argument("--core-masks", required=True, help="one per context, e.g. CORE_0,CORE_1,CORE_2 or AUTO")
    parser.add_argument("--cache", type=int, default=200, help="decoded images kept in RAM")
    parser.add_argument("--frames", type=int, default=1238, help="total frames processed")
    parser.add_argument("--out", required=True, type=Path)
    add_provenance_args(parser)
    args = parser.parse_args()

    masks = args.core_masks.split(",")
    for m in masks:
        if m not in CORE_MASK:
            parser.error(f"unknown core mask {m}")
    frames = [cv2.imread(str(p)) for p in list_images(args.images)[: args.cache]]
    contexts = [load_model(args.model, m) for m in masks]
    freq_before = frequency_state()

    for rknn, normalized in contexts:
        for _ in range(WARMUP):
            inp, meta = preprocess(frames[0])
            postprocess(rknn.inference(inputs=[inp])[0], meta, normalized)

    counter = itertools.count()
    lock = threading.Lock()
    latencies = [[] for _ in contexts]
    start = threading.Barrier(len(contexts) + 1)

    def worker(k: int) -> None:
        rknn, normalized = contexts[k]
        start.wait()
        while True:
            with lock:
                i = next(counter)
            if i >= args.frames:
                return
            t0 = time.perf_counter()
            inp, meta = preprocess(frames[i % len(frames)])
            postprocess(rknn.inference(inputs=[inp])[0], meta, normalized)
            latencies[k].append((time.perf_counter() - t0) * 1e3)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(contexts))]
    for t in threads:
        t.start()
    with ResourceMonitor() as monitor:
        start.wait()
        t_start = time.perf_counter()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t_start
    for rknn, _ in contexts:
        rknn.release()

    record = {
        **board_provenance(args, next(args.model.glob("*.rknn"))),
        "runtime": "rknnlite, python threads",
        "model": str(args.model),
        "contexts": len(contexts),
        "core_masks": masks,
        "images": args.images,
        "cached_images": len(frames),
        "frames": args.frames,
        "warmup_per_context": WARMUP,
        "measured_at": datetime.datetime.now().isoformat(),
        "system": system_info(),
        "frequency_before": freq_before,
        "frequency_after": frequency_state(),
        "wall_time_s": round(wall, 3),
        "throughput_fps": round(args.frames / wall, 2),
        "e2e_latency_ms": summarize([x for lat in latencies for x in lat]),
        "frames_per_context": [len(lat) for lat in latencies],
        "resource": monitor.result(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Done: {args.out}  throughput {record['throughput_fps']} FPS with {len(contexts)} context(s)")


if __name__ == "__main__":
    main()
