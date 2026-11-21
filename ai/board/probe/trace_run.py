"""Run a .rknn model on the NPU so that npu_trace.so can trace it.

The register words of a task do not depend on the pixel values, so the
input is seeded random uint8 data of the model input shape. Usage, on
the board (build the tracer first with `make build-npu-trace`):
    NPU_TRACE_OUT=<log> LD_PRELOAD=<npu_trace.so> python3 trace_run.py \
        --model <file.rknn> --core-mask CORE_0_1_2 --runs 2
"""

import argparse
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def main() -> None:
    """Load the model, run it --runs times on the given core mask."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--core-mask", default="CORE_0_1_2")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(args.model)) != 0:
        raise RuntimeError("load_rknn failed")
    mask = getattr(RKNNLite, "NPU_" + args.core_mask)
    if rknn.init_runtime(core_mask=mask) != 0:
        raise RuntimeError("init_runtime failed")
    rng = np.random.default_rng(args.seed)
    shape = (1, args.size, args.size, 3)
    inp = rng.integers(0, 256, size=shape, dtype=np.uint8)
    for _ in range(args.runs):
        outputs = rknn.inference(inputs=[inp])
    print("outputs", [o.shape for o in outputs])
    rknn.release()


if __name__ == "__main__":
    main()
