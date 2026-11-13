"""Raw NPU outputs for AI Quality, run ON THE BOARD with RKNNLite.

For each image: evaluator-identical input, RKNNLite inference, raw (4+nc, anchors) output.
Keeps only anchors with best score > EVAL_CONF (lossless for protocol v1 at conf 0.001), plus
each input's md5, checked by the PC evaluator. Called by evaluate.py --backend rknn over ssh.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ai.board.bench.bench_npu import load_model
from rk3588_common.model_io import EVAL_CONF, decode_split, preprocess
from rk3588_common.provenance import add_provenance_args, board_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="rknn_<precision>/ folder")
    parser.add_argument("--list", required=True, type=Path, help="text file, one image path per line")
    parser.add_argument("--out", required=True, type=Path)
    add_provenance_args(parser)
    args = parser.parse_args()

    images = [Path(p) for p in args.list.read_text().split()]
    rknn, normalized = load_model(args.model, "AUTO")
    split = bool(
        yaml.safe_load((args.model / "metadata.yaml").read_text()).get("split_head")
    )  # two raw outputs, decode on the CPU
    decode_s = []
    names, md5s, counts, idx, vals = [], [], [], [], []
    shape = None
    for path in images:
        inp, _ = preprocess(cv2.imread(str(path)))
        if split:
            raw = rknn.inference(inputs=[inp])
            t0 = time.perf_counter()
            out = decode_split(np.asarray(raw[0], dtype=np.float32), np.asarray(raw[1], dtype=np.float32))[0]
            decode_s.append(time.perf_counter() - t0)
        else:
            out = np.asarray(rknn.inference(inputs=[inp])[0], dtype=np.float32)[0]  # (4 + nc, anchors)
        shape = out.shape
        keep = np.flatnonzero(out[4:].max(0) > EVAL_CONF)
        names.append(path.name)
        md5s.append(hashlib.md5(np.ascontiguousarray(inp[0]).tobytes()).hexdigest())
        counts.append(len(keep))
        idx.append(keep.astype(np.int32))
        vals.append(out[:, keep].T)
    sdk_version = rknn.get_sdk_version()
    rknn.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        names=np.array(names),
        md5=np.array(md5s),
        counts=np.array(counts),
        idx=np.concatenate(idx),
        vals=np.concatenate(vals),
        shape=np.array(shape),
    )
    meta = {
        **board_provenance(args, next(args.model.glob("*.rknn"))),
        "normalized_boxes": normalized,
        "eval_conf": EVAL_CONF,
        "n_images": len(names),
        "kept_anchors": int(sum(counts)),
        "sdk_version": str(sdk_version),
        "split_head": split,
        "decode_ms_median": round(1000 * float(np.median(decode_s)), 3) if decode_s else None,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Done: {args.out} ({len(names)} images, {sum(counts)} anchors kept)")


if __name__ == "__main__":
    main()
