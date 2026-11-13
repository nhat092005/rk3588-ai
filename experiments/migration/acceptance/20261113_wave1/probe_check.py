"""Board check: deploy probe_infer (C++) against RKNNLite, same input.

Run on the board from ~/rk3588-board/ai:
    ../.venv/bin/python <this file> <rknn_dir> <images_dir> \
        ../deploy/bin/probe_infer
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from rknnlite.api import RKNNLite

from rk3588_common.model_io import (
    boxes_normalized,
    list_images,
    postprocess,
    preprocess,
)


def main() -> None:
    """Count images with identical raw outputs and detections."""
    model_dir, images_dir, probe = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    rknn_file = next(model_dir.glob("*.rknn"))
    metadata = yaml.safe_load((model_dir / "metadata.yaml").read_text())
    normalized = boxes_normalized(metadata)
    lite = RKNNLite()
    if lite.load_rknn(str(rknn_file)) or lite.init_runtime(
        core_mask=RKNNLite.NPU_CORE_AUTO
    ):
        raise SystemExit("RKNNLite init failed")
    n = raw_equal = det_equal = 0
    max_abs = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        raw_in, raw_out = Path(tmp) / "in.raw", Path(tmp) / "out.f32"
        for path in list_images(images_dir):
            inp, meta = preprocess(cv2.imread(str(path)))
            ref = np.asarray(lite.inference(inputs=[inp])[0], np.float32)
            raw_in.write_bytes(np.ascontiguousarray(inp).tobytes())
            subprocess.run(
                [probe, str(rknn_file), "1", str(raw_in), str(raw_out)],
                check=True,
                capture_output=True,
            )
            got = np.fromfile(raw_out, dtype=np.float32).reshape(ref.shape)
            n += 1
            raw_equal += np.array_equal(got, ref)
            max_abs = max(max_abs, float(np.abs(got - ref).max()))
            det_equal += np.array_equal(
                postprocess(got, meta, normalized),
                postprocess(ref, meta, normalized),
            )
    lite.release()
    print(
        f"images {n}, raw identical {raw_equal}, "
        f"detections identical {det_equal}, max |diff| {max_abs}"
    )


if __name__ == "__main__":
    main()
