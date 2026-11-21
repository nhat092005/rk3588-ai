# Submit trace of the ReLU model on 3 NPU cores

## Goal

Figure 16 (spatial partition of a convolution over the 3 NPU cores, with
halo rows) had no register evidence for 3x3 convolutions: the legacy
trace covers only the deployed SiLU model, where all 6 convolutions on
3 cores are 1x1. Question: when the model uses ReLU, does the NPU split a
3x3 convolution over the cores by rows, and does each core read halo rows?

## Method

- Board: Orange Pi 5 (RK3588), kernel 6.1.99-rockchip-rk3588, librknnrt
  2.3.2, rknpu driver 0.9.8 (printed by RKNN at load).
- Tracer: `ai/board/probe/npu_trace.c` of PR #11 (all register words of
  every task, no 400 word cut), built with
  `gcc -O2 -Wall -shared -fPIC -o npu_trace.so npu_trace.c -ldl -lpthread`.
- Runner: `ai/board/probe/trace_run.py`, 2 inferences, core mask
  `CORE_0_1_2`, 640x640 seeded random uint8 input (register words do not
  depend on pixel values).
- Command on the board:
  `NPU_TRACE_OUT=<log> LD_PRELOAD=<npu_trace.so> python3 trace_run.py --model <file.rknn> --core-mask CORE_0_1_2 --runs 2`
- Readers: `python3 -m ai.npu.latency.summarize|conv_split|diff_cores <log>`
  after `gunzip -k trace_*.log.gz`.

| Model | File | sha256 of the .rknn | Submits | Trace sha256 (uncompressed) |
|---|---|---|---|---|
| SiLU, deployed (control) | `best-rk3588.rknn` of `ppe5_yolov8n_baseline_100ep_s0` | `ea6002f6cbb11986...` | 154 | `b8b25d67136bda85...` |
| ReLU, decode in the graph | `yolov8n_relu_int8.rknn` | `6860ac8fd06e442d...` | 116 | `e100288b83a61a48...` |
| ReLU, split head (6 raw outputs) | `yolov8n_relu_raw_int8.rknn` | `4897e63666197588...` | 102 | `270a22a4da440540...` |

The two ReLU files were found in `~/tmp/a9c/models` on the board (file
date 2026-11-11); their build was not rechecked here. ReLU did not pass
the mAP gate of the operator support study, so it is not the deployed
model.

## Results (first inference, 3-core submits, mask 0x7)

Tasks per kind, from `results/*_summarize.txt`:

| Model | conv | elementwise | pool | conv on 1 core |
|---|---|---|---|---|
| SiLU | 6 | 52 | 29 | 144 + 44 |
| ReLU split head | 72 | 38 | 40 | 0 |
| ReLU, decode in graph | 72 | 46 | 39 | 40 + 44 |

Convolutions on 3 cores, from `results/*_conv_split.txt` (ReLU split head
counts, per inference):

| Kind | SiLU | ReLU |
|---|---|---|
| 1x1, split by rows, no halo | 6 | 26 |
| 3x3, split by rows with halo | 0 | 45 (38 checked by first row, 7 on the 640-wide input by row counts only) |
| 3x3, split by output channels (one feature address, 3 weight addresses) | 0 | 1 |

For the 3x3 convolutions split by rows (example, input width 20, output
rows 7, 7, 6):

- Input rows per core 8, 9, 7: output rows + 1 for the first and last
  core, + 2 for the middle core.
- First input row per core 0, 6, 13: the core starts one row before its
  first output row (7 and 14), so neighbouring cores read the same row.
- `pad_top` is 1 on core 0 only. The other cores get the border from the
  halo row.
- Stride-2 3x3 convolutions follow the same pattern with the start row at
  2 x first output row - 1.
- The weight address is the same on the 3 cores.

For 1x1 convolutions input rows equal output rows and the first input
rows are 0, 7, 14: disjoint strips, no halo, same as the SiLU control.

Pool tasks (en 0x60) differ between the cores only in the PC chain
address (`diff_cores`: ndiff 1 for most, 0 for 2 tasks), in the ReLU
traces as in the SiLU one.

## Conclusion

- Measured: the ReLU model splits 3x3 convolutions over 3 cores by
  rows with a halo of 1 row at each inner border (38 of its 46 3x3 tasks
  per inference checked on the start rows, 7 more by row counts), and one
  3x3 by output channels.
  The SiLU model does not (0 of its 3x3 tasks on 3 cores). The plan in the
  WorkLoad log (ReLU `ConvRelu` 51/51 split, SiLU `ConvExSwish` 0/57)
  matches the registers.
- Figure 16 is right for a ReLU model (halo bands on a 3x3 convolution)
  and wrong for the deployed SiLU model, where the figure would have to
  show a 1x1 convolution without halo.

## Not determined

- Why SiLU convolutions are not split: not visible in the registers.
- What the pool tasks do on 3 cores: the register words are identical on
  the three cores, so a row split is not visible; whether each core pools
  the same data is not known.
- `first_row` of the 640-wide convolutions (3-channel network input):
  the 16-byte channel group assumption does not hold, those numbers are
  not read.
- One model of each activation, one input size (640x640), one core mask.

## Evidence

`conv_split` on the first inference, ReLU split head (counts per kind of
task):

```
    8 x kernel=1x1 in_w=20 in_h=(7, 7, 6) out_h=(7, 7, 6) pad_top=(0, 0, 0) first_row=(0, 7, 14) weights=1
    8 x kernel=1x1 in_w=40 in_h=(14, 13, 13) out_h=(14, 13, 13) pad_top=(0, 0, 0) first_row=(0, 14, 27) weights=1
    1 x kernel=1x1 in_w=80 in_h=(13, 13, 13) out_h=(13, 13, 13) pad_top=(0, 0, 0) first_row=(0, 13, 26) weights=1
    1 x kernel=1x1 in_w=80 in_h=(14, 14, 13) out_h=(14, 14, 13) pad_top=(0, 0, 0) first_row=(0, 14, 28) weights=1
    5 x kernel=1x1 in_w=80 in_h=(27, 27, 26) out_h=(27, 27, 26) pad_top=(0, 0, 0) first_row=(0, 27, 54) weights=1
    1 x kernel=1x1 in_w=160 in_h=(27, 26, 26) out_h=(27, 26, 26) pad_top=(0, 0, 0) first_row=(0, 27, 53) weights=1
    1 x kernel=1x1 in_w=160 in_h=(27, 27, 27) out_h=(27, 27, 27) pad_top=(0, 0, 0) first_row=(0, 27, 54) weights=1
    1 x kernel=1x1 in_w=160 in_h=(54, 53, 53) out_h=(54, 53, 53) pad_top=(0, 0, 0) first_row=(0, 54, 107) weights=1
    8 x kernel=3x3 in_w=20 in_h=(8, 9, 7) out_h=(7, 7, 6) pad_top=(1, 0, 0) first_row=(0, 6, 13) weights=1
    1 x kernel=3x3 in_w=40 in_h=(14, 15, 13) out_h=(7, 7, 6) pad_top=(1, 0, 0) first_row=(0, 13, 27) weights=1
   12 x kernel=3x3 in_w=40 in_h=(15, 15, 14) out_h=(14, 13, 13) pad_top=(1, 0, 0) first_row=(0, 13, 26) weights=1
    1 x kernel=3x3 in_w=40 in_h=(40, 40, 40) out_h=(20, 20, 20) pad_top=(1, 1, 1) first_row=(0, 0, 0) weights=3
    2 x kernel=3x3 in_w=80 in_h=(28, 27, 27) out_h=(14, 13, 13) pad_top=(1, 0, 0) first_row=(0, 27, 53) weights=1
   10 x kernel=3x3 in_w=80 in_h=(28, 29, 27) out_h=(27, 27, 26) pad_top=(1, 0, 0) first_row=(0, 26, 53) weights=1
    1 x kernel=3x3 in_w=160 in_h=(54, 55, 53) out_h=(27, 27, 26) pad_top=(1, 0, 0) first_row=(0, 53, 107) weights=1
    2 x kernel=3x3 in_w=160 in_h=(55, 55, 54) out_h=(54, 53, 53) pad_top=(1, 0, 0) first_row=(0, 53, 106) weights=1
    1 x kernel=3x3 in_w=320 in_h=(54, 55, 55) out_h=(27, 27, 27) pad_top=(1, 0, 0) first_row=(0, 53, 107) weights=1
    1 x kernel=3x3 in_w=320 in_h=(55, 53, 53) out_h=(27, 26, 26) pad_top=(0, 0, 0) first_row=(0, 54, 106) weights=1
    5 x kernel=3x3 in_w=640 in_h=(31, 31, 31) out_h=(15, 15, 15) pad_top=(0, 0, 0) first_row=(0, None, None) weights=1
    1 x kernel=3x3 in_w=640 in_h=(32, 33, 33) out_h=(16, 16, 16) pad_top=(1, 0, 0) first_row=(0, None, None) weights=1
    1 x kernel=3x3 in_w=640 in_h=(33, 33, 31) out_h=(16, 16, 15) pad_top=(0, 0, 0) first_row=(0, 6, 12) weights=1
```

Same on the SiLU control (all 3-core convs):

```
    2 x kernel=1x1 in_w=20 in_h=(7, 7, 6) out_h=(7, 7, 6) pad_top=(0, 0, 0) first_row=(0, 7, 14) weights=1
    2 x kernel=1x1 in_w=40 in_h=(14, 13, 13) out_h=(14, 13, 13) pad_top=(0, 0, 0) first_row=(0, 14, 27) weights=1
    2 x kernel=1x1 in_w=80 in_h=(27, 27, 26) out_h=(27, 27, 26) pad_top=(0, 0, 0) first_row=(0, 27, 54) weights=1
```

## Files

Only this note is tracked. The traces and the reader outputs stay local
(ignored by git, `experiments/**/*.log.gz` and `experiments/npu/**/results/`):

- `trace_silu.log.gz`, `trace_relu.log.gz`, `trace_relu_raw.log.gz`:
  gzip of the three logs. Rerun the readers on them as in Method.
- `results/*_summarize.txt`, `results/*_conv_split.txt`: reader outputs.

Full sha256 of the uncompressed traces:

```
b8b25d67136bda858e3190c41f5d5c7f2bfbfef769081fd88d5f80caf8821912  trace_silu.log
e100288b83a61a48774e984a55157c528a6cc30612189d22c61eb0f8acfb4da3  trace_relu.log
270a22a4da44054057e7e26838a6be6ee57225f8280ef599b414140d0a5c8225  trace_relu_raw.log
```
