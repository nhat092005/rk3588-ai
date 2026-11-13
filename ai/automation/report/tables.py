"""CLI: build every report table (A-M) from result files.

Run: make tables DATASET=<dataset>
Writes results/<dataset>/table_<X>_*.md, tables.csv, and SOURCES.md
(source file/field per cell). Missing data shows [TBD: ...];
consistency checks (protocol, fingerprint, sha256, freq lock) are
listed in SOURCES.md and exit 1 on failure. Mean ± std uses sample
std; hardware tables use seed-0.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from ai.core.dataset import load_dataset_yaml
from ai.core.evaluator import EVAL_SETS, EVAL_SETS_BY_DATASET, PROTOCOL_VERSION
from ai.paths import LEADERBOARDS_DIR, REPO_ROOT, RESULTS_DIR, RUNS_DIR

# Set in main() from the class names of the dataset yaml.
CLASSES: list[str] = []
HW_SEED = "0"
MASKS_1CTX = ["AUTO", "CORE_0", "CORE_0_1", "CORE_0_1_2"]
MASKS_3CTX = "CORE_0,CORE_1,CORE_2"


class Builder:
    """Collect table cells, sources, warnings, and check errors."""

    def __init__(self, dataset: str) -> None:
        """Initialize the table builder for the given dataset."""
        self.dataset = dataset
        self.cells: list[dict] = []  # table, row, column, value, source
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self.json_cache: dict[Path, dict | None] = {}

    # ---- sources -----------------------------------------------------
    def load(self, path: Path) -> dict | None:
        """Load and validate a JSON result file with caching."""
        if path not in self.json_cache:
            data = json.loads(path.read_text()) if path.exists() else None
            if data is not None:
                self.check_file(path, data)
            self.json_cache[path] = data
        return self.json_cache[path]

    def check_file(self, path: Path, data: dict) -> None:
        """Record protocol, dirty-repo, and frequency-lock issues."""
        rel = path.relative_to(REPO_ROOT)
        if (
            "protocol_version" in data
            and data["protocol_version"] != PROTOCOL_VERSION
        ):
            self.problems.append(
                f"{rel}: protocol {data['protocol_version']} != "
                f"{PROTOCOL_VERSION}"
            )
        dirty = sorted(
            name
            for name, state in data.get("repos", {}).items()
            if state.get("dirty")
        )
        if dirty:
            self.warnings.append(
                f"{rel}: produced with uncommitted code in {', '.join(dirty)}"
            )
        freq = data.get("frequency_before")
        if freq is not None and not freq.get("locked"):
            self.problems.append(
                f"{rel}: frequencies not locked during the measurement "
                "(on the board: sudo -n /usr/local/sbin/lock_freq.sh lock)"
            )

    def cell(
        self,
        table: str,
        row: str,
        col: str,
        path: Path,
        keys: list,
        fmt: str = "{:.4f}",
    ) -> str:
        """Read one cell from a JSON file and record its provenance."""
        data = self.load(path)
        rel = path.relative_to(REPO_ROOT)
        if data is None:
            value, text = None, f"[TBD: missing {rel}]"
        else:
            value = data
            for k in keys:
                value = value.get(k) if isinstance(value, dict) else None
            text = (
                "[TBD: no field " + ".".join(map(str, keys)) + "]"
                if value is None
                else (
                    fmt.format(value)
                    if isinstance(value, (int, float))
                    else str(value)
                )
            )
        self.cells.append(
            {
                "table": table,
                "row": row,
                "column": col,
                "value": value if value is not None else "",
                "source": f"{rel}:{'.'.join(map(str, keys))}",
            }
        )
        return text

    def agg(
        self, table: str, row: str, col: str, values: list[tuple[float, str]]
    ) -> str:
        """Mean ± std over seeds; values = [(value, source)]."""
        nums = [v for v, _ in values if v is not None]
        src = "; ".join(s for _, s in values)
        if not nums:
            text, value = "[TBD]", ""
        elif len(nums) == 1:
            text, value = f"{nums[0]:.4f} (1 seed)", nums[0]
        else:
            m, sd = statistics.mean(nums), statistics.stdev(nums)
            text, value = f"{m:.4f} ± {sd:.4f}", f"{m} ± {sd}"
        self.cells.append(
            {
                "table": table,
                "row": row,
                "column": col,
                "value": value,
                "source": src,
            }
        )
        return text

    # ---- runs --------------------------------------------------------
    def leaderboard(self) -> list[dict]:
        """Return leaderboard rows matching the current protocol."""
        path = LEADERBOARDS_DIR / f"{self.dataset}.csv"
        if not path.exists():
            return []
        with path.open() as fh:
            return [
                r
                for r in csv.DictReader(fh)
                if r["protocol"] == PROTOCOL_VERSION
            ]

    def runs_by_model(self) -> dict[str, dict[str, str]]:
        """Map model -> seed -> run_id from the leaderboard."""
        out: dict[str, dict[str, str]] = defaultdict(dict)
        for r in self.leaderboard():
            epochs = re.search(r"_(\d+)ep_s\d+$", r["run_id"]).group(1)
            # A 200-epoch run of the same model and seed is its own row,
            # not a replacement of the 100-epoch run.
            out[r["model"] if epochs == "100" else f"{r['model']}_{epochs}ep"][
                r["seed"]
            ] = r["run_id"]
        return dict(sorted(out.items()))

    def fingerprint_check(self, eval_set: str, files: list[Path]) -> None:
        """Check that eval files for an eval_set share a fingerprint."""
        fps = {
            self.load(f)["dataset_fingerprint"] for f in files if self.load(f)
        }
        if len(fps) > 1:
            self.problems.append(
                f"{eval_set}: {len(fps)} different dataset fingerprints "
                "across eval files"
            )

    def sha_check(
        self, run_id: str, precision: str, paths: list[Path]
    ) -> None:
        """Verify eval/bench RKNN sha256 matches export metadata."""
        export = self.load(RUNS_DIR / run_id / f"export_{precision}.json")
        if not export:
            return
        if "rknn_sha256" not in export:
            self.problems.append(
                f"runs/{run_id}/export_{precision}.json has no rknn_sha256 "
                "(old export: re-run make export-npu)"
            )
            return
        for p in paths:
            d = self.load(p)
            if (
                d
                and d.get("model_sha256")
                and d["model_sha256"] != export["rknn_sha256"]
            ):
                self.problems.append(
                    f"{p.relative_to(REPO_ROOT)}: model sha256 differs "
                    f"from export_{precision}.json"
                )


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Format a header and rows as a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ---- tables ----------------------------------------------------------
def table_a(b: Builder) -> str:
    """Build Table A (Dataset Summary)."""
    data = load_dataset_yaml(b.dataset)
    root = Path(data["path"])
    rows = []
    for split in ("train", "val", "test"):
        labels = sorted((root / "labels" / split).glob("*.txt"))
        n_inst = sum(
            len([ln for ln in p.read_text().splitlines() if ln.strip()])
            for p in labels
        )
        rows.append([split, str(len(labels)), str(n_inst)])
        rel_split = (root / "labels" / split).relative_to(REPO_ROOT)
        b.cells.append(
            {
                "table": "A",
                "row": split,
                "column": "images/instances",
                "value": f"{len(labels)}/{n_inst}",
                "source": f"{rel_split}/*.txt",
            }
        )
    out = [
        f"# Table A. Dataset Summary: `{b.dataset}`",
        "",
        "Counted from the label files.",
        "",
        md_table(["Split", "#Images", "#Instances"], rows),
    ]
    for eval_set in EVAL_SETS_BY_DATASET[b.dataset]:
        ds, split = EVAL_SETS[eval_set]
        ds_data = load_dataset_yaml(ds)
        droot = Path(ds_data["path"])
        split_path = droot / ds_data[split]
        if split_path.is_dir():
            stems = [p.stem for p in split_path.iterdir()]
        else:  # txt list
            stems = [Path(ln).stem for ln in split_path.read_text().split()]
        counts = [0] * len(CLASSES)
        # images/<split> and labels/<split> share the split name.
        label_dir = (
            droot
            / "labels"
            / (split_path.name if split_path.is_dir() else "test")
        )
        for stem in stems:
            f = label_dir / f"{stem}.txt"
            for ln in f.read_text().splitlines() if f.exists() else []:
                if ln.strip():
                    counts[int(ln.split()[0])] += 1
        total = sum(counts)
        crow = [
            [c, str(n), f"{100 * n / total:.1f}%"]
            for c, n in zip(CLASSES, counts, strict=True)
        ] + [["**Total**", f"**{total}**", "100%"]]
        out += [
            "",
            f"## Class distribution, eval set `{eval_set}` "
            f"({len(stems)} images)",
            "",
            md_table(["Class", "Instances", "%"], crow),
        ]
    return "\n".join(out)


def table_b(b: Builder, runs: dict) -> str:
    """Build Table B (Training Configuration)."""
    keys = [
        "epochs",
        "batch",
        "imgsz",
        "optimizer",
        "lr0",
        "momentum",
        "warmup_bias_lr",
        "weight_decay",
        "seed",
    ]
    rows = []
    for model, seeds in runs.items():
        for _seed, run_id in sorted(seeds.items()):
            args_path = RUNS_DIR / run_id / "args.yaml"
            args = (
                yaml.safe_load(args_path.read_text())
                if args_path.exists()
                else {}
            )
            ev = (
                b.load(
                    RUNS_DIR
                    / run_id
                    / "eval"
                    / f"pt_fp32_{EVAL_SETS_BY_DATASET[b.dataset][0]}.json"
                )
                or {}
            )
            vals = [str(args.get(k, "[TBD]")) for k in keys] + [
                ev.get("ultralytics_version", "[TBD]"),
                ev.get("torch_version", "[TBD]"),
            ]
            rows.append([model, run_id, *vals])
            # vals also holds the two versions; cells only for keys.
            for k, v in zip(keys, vals, strict=False):
                b.cells.append(
                    {
                        "table": "B",
                        "row": run_id,
                        "column": k,
                        "value": v,
                        "source": f"{args_path.relative_to(REPO_ROOT)}:{k}",
                    }
                )
    return "\n".join(
        [
            f"# Table B. Training Configuration: `{b.dataset}`",
            "",
            "From `runs/<run_id>/args.yaml` (written by Ultralytics) "
            "and `eval/pt_fp32_*.json`.",
            "",
            md_table(["Model", "run_id", *keys, "ultralytics", "torch"], rows)
            if rows
            else "[TBD: no run in the leaderboard]",
        ]
    )


def table_c(b: Builder, runs: dict) -> str:
    """Build Table C (Hardware / Software Environment)."""
    candidates = sorted(RUNS_DIR.glob("*/bench/tier2_*_AUTO.json"))
    candidates = [
        p
        for p in candidates
        if p.parts[-3] in {r for s in runs.values() for r in s.values()}
    ]
    if not candidates:
        return (
            "# Table C. Hardware / Software Environment\n\n"
            "[TBD: missing runs/<run_id>/bench/tier2_<prec>_AUTO.json "
            "(make bench-board)]"
        )
    src = candidates[-1]
    rows = [
        [f"system.{k}", b.cell("C", k, "value", src, ["system", k], "{}")]
        for k in ("os", "kernel", "python", "ram_total_mib", "cpu_count")
    ]
    sdk = " ".join(
        re.findall(
            r"(?:API|DRV): \S+",
            b.cell("C", "sdk", "value", src, ["sdk_version"], "{}"),
        )
    )
    rows.append(
        ["sdk_version (RKNN-Toolkit-Lite2 API, NPU driver)", sdk or "[TBD]"]
    )
    rows.extend(
        [
            f"governor {name}",
            b.cell(
                "C",
                name,
                "governor",
                src,
                ["frequency_before", name, "governor"],
                "{}",
            ),
        ]
        for name in ("policy0", "policy4", "policy6", "npu", "ddr")
    )
    rows.append(
        [
            "frequency locked",
            b.cell(
                "C",
                "locked",
                "value",
                src,
                ["frequency_before", "locked"],
                "{}",
            ),
        ]
    )
    rows.append(
        [
            "NPU thermal zone",
            b.cell(
                "C",
                "thermal",
                "value",
                src,
                ["resource", "thermal_zone"],
                "{}",
            ),
        ]
    )
    return "\n".join(
        [
            "# Table C. Hardware / Software Environment",
            "",
            f"From `{src.relative_to(REPO_ROOT)}` (latest tier-2 "
            "measurement). Fixed rows (board, camera): Table C in the Guide.",
            "",
            md_table(["Item", "Value"], rows),
        ]
    )


def eval_metrics(
    b: Builder, run_id: str, tag: str, eval_set: str
) -> tuple[Path, dict | None]:
    """Load an eval metrics file for a run, backend tag, and split."""
    path = RUNS_DIR / run_id / "eval" / f"{tag}_{eval_set}.json"
    return path, b.load(path)


def table_d(b: Builder, runs: dict) -> str:
    """Build Table D (Baseline Accuracy FP32 PyTorch)."""
    out = [
        f"# Table D. Baseline Accuracy (FP32 PyTorch): `{b.dataset}`",
        "",
        "mean ± std over seeds (a single seed shows `(1 seed)`); "
        "AP = per-class AP50-95.",
        "",
    ]
    cols = ["precision", "recall", "map50", "map50_95"] + [
        f"AP {c}" for c in CLASSES
    ]
    for eval_set in EVAL_SETS_BY_DATASET[b.dataset]:
        rows, files = [], []
        for model, seeds in runs.items():
            per_seed = [
                eval_metrics(b, rid, "pt_fp32", eval_set)
                for rid in seeds.values()
            ]
            files += [p for p, _ in per_seed]
            row = [model, ",".join(sorted(seeds))]
            for col in cols:
                vals = []
                for p, d in per_seed:
                    rel_p = p.relative_to(REPO_ROOT)
                    if d is None:
                        vals.append((None, f"[missing {rel_p}]"))
                    elif col.startswith("AP "):
                        c = col[3:]
                        vals.append(
                            (
                                d["per_class"].get(c, {}).get("ap50_95"),
                                f"{rel_p}:per_class.{c}.ap50_95",
                            )
                        )
                    else:
                        vals.append((d[col], f"{rel_p}:{col}"))
                row.append(b.agg("D", f"{model}/{eval_set}", col, vals))
            rows.append(row)
        b.fingerprint_check(eval_set, files)
        out += [
            f"## Eval set `{eval_set}`",
            "",
            md_table(["Model", "Seeds", *cols], rows),
            "",
        ]
    return "\n".join(out)


def table_e(b: Builder, runs: dict) -> str:
    """Build Table E (Model Complexity)."""
    rows = []
    for model, seeds in runs.items():
        path = RUNS_DIR / seeds.get(HW_SEED, "?") / "complexity.json"
        rows.append(
            [model]
            + [
                b.cell("E", model, k, path, keys, f)
                for k, keys, f in (
                    ("Params (M)", ["params_m"], "{:.3f}"),
                    ("GFLOPs", ["gflops"], "{:.3f}"),
                    ("MACs (G)", ["macs_g"], "{:.3f}"),
                    ("ONNX FP32 (MB)", ["onnx_fp32_mb"], "{:.3f}"),
                    ("RKNN FP16 (MB)", ["rknn_mb", "rknn_fp16"], "{:.3f}"),
                    ("RKNN INT8 (MB)", ["rknn_mb", "rknn_int8"], "{:.3f}"),
                )
            ]
        )
    return "\n".join(
        [
            f"# Table E. Model Complexity: `{b.dataset}`",
            "",
            "Seed-0 run; fused model, 640×640, MACs = GFLOPs / 2 "
            "(make complexity).",
            "",
            md_table(
                [
                    "Model",
                    "Params (M)",
                    "GFLOPs",
                    "MACs (G)",
                    "ONNX FP32 (MB)",
                    "RKNN FP16 (MB)",
                    "RKNN INT8 (MB)",
                ],
                rows,
            ),
        ]
    )


def table_f(b: Builder, runs: dict) -> str:
    """Build Table F (Quantization Impact)."""
    out = [
        f"# Table F. Quantization Impact: `{b.dataset}`",
        "",
        "Δ = FP32 − NPU (positive = accuracy lost), mean ± std over seeds; "
        "AP = AP50-95.",
        "",
    ]
    for eval_set in EVAL_SETS_BY_DATASET[b.dataset]:
        rows = []
        for model, seeds in runs.items():
            # Hybrid INT8 standard (recipe h5).
            h5 = (
                ("int8h5",)
                if all(
                    (RUNS_DIR / rid / "export_int8h5.json").exists()
                    for rid in seeds.values()
                )
                else ()
            )
            for prec in ("fp32", "fp16", "int8", *h5):
                tag = "pt_fp32" if prec == "fp32" else f"rknn_{prec}"
                row = [model, prec.upper()]
                maps = []
                deltas: dict[str, list] = {
                    c: [] for c in ["map50_95", *CLASSES]
                }
                for rid in seeds.values():
                    fp_path, fp = eval_metrics(b, rid, "pt_fp32", eval_set)
                    q_path, q = eval_metrics(b, rid, tag, eval_set)
                    if tag != "pt_fp32":
                        b.sha_check(rid, prec, [q_path])
                    rel = q_path.relative_to(REPO_ROOT)
                    maps.append(
                        (q["map50_95"] if q else None, f"{rel}:map50_95")
                    )
                    for c, c_deltas in deltas.items():
                        if prec == "fp32" or fp is None or q is None:
                            continue
                        rel_fp = fp_path.relative_to(REPO_ROOT)
                        if (
                            fp["dataset_fingerprint"]
                            != q["dataset_fingerprint"]
                        ):
                            b.problems.append(
                                f"{rel}: dataset fingerprint differs from "
                                f"{rel_fp}"
                            )
                        a = (
                            fp["map50_95"]
                            if c == "map50_95"
                            else fp["per_class"].get(c, {}).get("ap50_95")
                        )
                        z = (
                            q["map50_95"]
                            if c == "map50_95"
                            else q["per_class"].get(c, {}).get("ap50_95")
                        )
                        c_deltas.append(
                            (
                                a - z
                                if a is not None and z is not None
                                else None,
                                f"{rel_fp} - {rel}",
                            )
                        )
                row.append(
                    b.agg("F", f"{model}/{prec}/{eval_set}", "map50_95", maps)
                )
                for c, c_deltas in deltas.items():
                    row.append(
                        "ref"
                        if prec == "fp32"
                        else b.agg(
                            "F",
                            f"{model}/{prec}/{eval_set}",
                            f"delta_{c}",
                            c_deltas,
                        )
                    )
                rows.append(row)
        out += [
            f"## Eval set `{eval_set}`",
            "",
            md_table(
                ["Model", "Precision", "mAP50-95", "ΔmAP50-95"]
                + [f"ΔAP {c}" for c in CLASSES],
                rows,
            ),
            "",
        ]
    return "\n".join(out)


def table_g(b: Builder, runs: dict) -> str:
    """Build Table G (Latency Breakdown)."""
    rows = []
    for model, seeds in runs.items():
        rid = seeds.get(HW_SEED, "?")
        for prec in ("fp16", "int8"):
            t1 = RUNS_DIR / rid / "bench" / f"tier1_{prec}_AUTO.json"
            t2 = RUNS_DIR / rid / "bench" / f"tier2_{prec}_AUTO.json"
            b.sha_check(rid, prec, [t1, t2])
            r = f"{model}/{prec}"
            rows.append(
                [
                    model,
                    prec.upper(),
                    b.cell(
                        "G",
                        r,
                        "pre_p50",
                        t2,
                        ["latency_ms", "pre", "p50"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "G", r, "npu_tier1", t1, ["npu_latency_ms"], "{:.2f}"
                    ),
                    b.cell(
                        "G",
                        r,
                        "npu_tier2_p50",
                        t2,
                        ["latency_ms", "npu", "p50"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "G",
                        r,
                        "post_p50",
                        t2,
                        ["latency_ms", "post", "p50"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "G",
                        r,
                        "e2e_mean",
                        t2,
                        ["latency_ms", "e2e", "mean"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "G",
                        r,
                        "e2e_p50",
                        t2,
                        ["latency_ms", "e2e", "p50"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "G",
                        r,
                        "e2e_p95",
                        t2,
                        ["latency_ms", "e2e", "p95"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "G", r, "fps_latency", t2, ["fps_latency"], "{:.1f}"
                    ),
                ]
            )
    return "\n".join(
        [
            f"# Table G. Latency Breakdown (ms): `{b.dataset}`",
            "",
            "NPU (tier 1) = median of 5 `eval_perf` calls; other columns "
            "from tier 2 on the board (`css_test`, core_mask AUTO). "
            "FPS_latency = 1000 / E2E mean.",
            "",
            md_table(
                [
                    "Model",
                    "Precision",
                    "Pre P50",
                    "NPU (tier 1)",
                    "NPU P50 (tier 2)",
                    "Post P50",
                    "E2E mean",
                    "E2E P50",
                    "E2E P95",
                    "FPS_latency",
                ],
                rows,
            ),
        ]
    )


def table_h(b: Builder, runs: dict) -> str:
    """Build Table H (Hardware Comparison)."""
    rows = []
    for model, seeds in runs.items():
        rid = seeds.get(HW_SEED, "?")
        bench, ev = RUNS_DIR / rid / "bench", RUNS_DIR / rid / "eval"
        test = EVAL_SETS_BY_DATASET[b.dataset][0]
        for device, bfile, efile in (
            (
                "CPU board, ONNX FP32 (4 thread)",
                bench / "tier2_cpu_onnx_t4.json",
                ev / f"pt_fp32_{test}.json",
            ),
            (
                "NPU board, RKNN FP16",
                bench / "tier2_fp16_AUTO.json",
                ev / f"rknn_fp16_{test}.json",
            ),
            (
                "NPU board, RKNN INT8",
                bench / "tier2_int8_AUTO.json",
                ev / f"rknn_int8_{test}.json",
            ),
            (
                "PC GPU (reference), PyTorch FP32",
                bench / "gpu_pt_fp32.json",
                ev / f"pt_fp32_{test}.json",
            ),
        ):
            r = f"{model}/{device}"
            rows.append(
                [
                    model,
                    device,
                    b.cell("H", r, "map50_95", efile, ["map50_95"]),
                    b.cell(
                        "H",
                        r,
                        "e2e_p50",
                        bfile,
                        ["latency_ms", "e2e", "p50"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "H",
                        r,
                        "e2e_p95",
                        bfile,
                        ["latency_ms", "e2e", "p95"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "H", r, "fps_latency", bfile, ["fps_latency"], "{:.1f}"
                    ),
                ]
            )
    return "\n".join(
        [
            f"# Table H. Hardware Comparison: `{b.dataset}`",
            "",
            "CPU ONNX mAP50-95 is the PyTorch FP32 value (same weights; "
            "ONNX = PyTorch is checked by `make evaluate BACKEND=onnx`).",
            "",
            md_table(
                [
                    "Model",
                    "Device / Precision",
                    "mAP50-95",
                    "E2E P50 (ms)",
                    "E2E P95 (ms)",
                    "FPS_latency",
                ],
                rows,
            ),
        ]
    )


def table_i(b: Builder, runs: dict) -> str:
    """Build Table I (Deployment Configuration INT8)."""
    rows = []
    for model, seeds in runs.items():
        bench = RUNS_DIR / seeds.get(HW_SEED, "?") / "bench"
        for mask in [*MASKS_1CTX, MASKS_3CTX]:
            thr = bench / f"throughput_int8_{mask.replace(',', '-')}.json"
            r = f"{model}/{mask}"
            npu = (
                b.cell(
                    "I",
                    r,
                    "npu_ms",
                    bench / f"tier1_int8_{mask}.json",
                    ["npu_latency_ms"],
                    "{:.2f}",
                )
                if "," not in mask
                else "-"
            )
            rows.append(
                [
                    model,
                    f"`{mask}`",
                    "3" if "," in mask else "1",
                    npu,
                    b.cell(
                        "I",
                        r,
                        "throughput_fps",
                        thr,
                        ["throughput_fps"],
                        "{:.1f}",
                    ),
                    b.cell(
                        "I",
                        r,
                        "avg_cpu",
                        thr,
                        ["resource", "avg_system_cpu_pct"],
                        "{:.1f}",
                    ),
                    b.cell(
                        "I",
                        r,
                        "temp_end",
                        thr,
                        ["resource", "temp_end_c"],
                        "{:.1f}",
                    ),
                ]
            )
    return "\n".join(
        [
            f"# Table I. Deployment Configuration (INT8): `{b.dataset}`",
            "",
            "NPU latency from tier 1; throughput, system CPU (%) and NPU "
            "temperature from `bench_throughput`.",
            "",
            md_table(
                [
                    "Model",
                    "core_mask",
                    "Contexts",
                    "NPU latency (ms)",
                    "Throughput FPS",
                    "Avg CPU (%)",
                    "Temp end (°C)",
                ],
                rows,
            ),
        ]
    )


def table_k(b: Builder, runs: dict, top: int = 20) -> str:
    """Build Table K (Operator / Layer Bottleneck Profile)."""
    out = [
        "# Table K. Operator / Layer Bottleneck Profile (INT8, AUTO): "
        f"`{b.dataset}`",
        "",
        "From the per-layer table of `eval_perf(perf_debug=True)` (debug "
        "mode, times may exceed real ones). Multi-core support: see Table "
        "K in the Guide.",
        "",
    ]
    for model, seeds in runs.items():
        path = (
            RUNS_DIR
            / seeds.get(HW_SEED, "?")
            / "bench"
            / "tier1_int8_AUTO_layers.json"
        )
        data = b.load(path)
        if data is None:
            out += [
                f"## {model}",
                "",
                f"[TBD: missing {path.relative_to(REPO_ROOT)}]",
                "",
            ]
            continue
        layers = data["layers"]
        total = sum(r["time_us"] or 0 for r in layers)
        by_op = defaultdict(lambda: [0, 0, set()])
        for r in layers:
            by_op[r["op"]][0] += 1
            by_op[r["op"]][1] += r["time_us"] or 0
            by_op[r["op"]][2].add(r["target"])
        op_rows = [
            [
                op,
                str(n),
                "/".join(sorted(t)),
                str(us),
                f"{100 * us / total:.1f}%",
            ]
            for op, (n, us, t) in sorted(
                by_op.items(), key=lambda kv: -kv[1][1]
            )
        ]
        top_rows = [
            [
                str(r["id"]),
                r["op"],
                r["target"],
                str(r["time_us"]),
                f"{100 * (r['time_us'] or 0) / total:.1f}%",
                f"{r['ddr_cycles']}/{r['npu_cycles']}",
                "DDR > NPU"
                if (r["ddr_cycles"] or 0) > (r["npu_cycles"] or 0)
                else "",
                r["name"],
            ]
            for r in sorted(layers, key=lambda r: -(r["time_us"] or 0))[:top]
        ]
        b.cells.append(
            {
                "table": "K",
                "row": model,
                "column": "total_us",
                "value": total,
                "source": f"{path.relative_to(REPO_ROOT)}:layers[*].time_us",
            }
        )
        out += [
            f"## {model}: by op type (total {total} µs, {len(layers)} layers)",
            "",
            md_table(
                ["Op", "Layers", "Runs on", "Time (µs)", "% of total"], op_rows
            ),
            "",
            f"## {model}: {top} slowest layers",
            "",
            md_table(
                [
                    "ID",
                    "Op",
                    "Runs on",
                    "µs",
                    "%",
                    "DDR/NPU cycles",
                    "Memory-bound?",
                    "Name",
                ],
                top_rows,
            ),
            "",
        ]
    return "\n".join(out)


def table_l(b: Builder, runs: dict) -> str:
    """Build Table L (Hardware Resource Usage)."""
    rows = []
    for model, seeds in runs.items():
        bench = RUNS_DIR / seeds.get(HW_SEED, "?") / "bench"
        for prec in ("fp16", "int8"):
            t2, t1 = (
                bench / f"tier2_{prec}_AUTO.json",
                bench / f"tier1_{prec}_AUTO.json",
            )
            r = f"{model}/{prec}"
            rows.append(
                [
                    model,
                    prec.upper(),
                    b.cell(
                        "L",
                        r,
                        "peak_rss",
                        t2,
                        ["resource", "peak_rss_mib"],
                        "{:.1f}",
                    ),
                    b.cell(
                        "L",
                        r,
                        "npu_mem",
                        t1,
                        ["npu_memory_mib", "total_memory"],
                        "{:.2f}",
                    ),
                    b.cell(
                        "L",
                        r,
                        "avg_cpu_process",
                        t2,
                        ["resource", "avg_process_cpu_pct"],
                        "{:.1f}",
                    ),
                    b.cell(
                        "L",
                        r,
                        "temp_start",
                        t2,
                        ["resource", "temp_start_c"],
                        "{:.1f}",
                    ),
                    b.cell(
                        "L",
                        r,
                        "temp_end",
                        t2,
                        ["resource", "temp_end_c"],
                        "{:.1f}",
                    ),
                ]
            )
    return "\n".join(
        [
            f"# Table L. Hardware Resource Usage: `{b.dataset}`",
            "",
            "Tier 2 (core_mask AUTO). CPU = process % (100 = one core). "
            "NPU memory = `eval_memory` total (tier 1).",
            "",
            md_table(
                [
                    "Model",
                    "Precision",
                    "Peak RAM VmRSS (MiB)",
                    "NPU memory (MiB)",
                    "Avg CPU process (%)",
                    "Temp start (°C)",
                    "Temp end (°C)",
                ],
                rows,
            ),
        ]
    )


def table_j_m(b: Builder) -> str:
    """Build placeholder Table J, M."""
    return "\n".join(
        [
            "# Table J, M",
            "",
            "Baseline rows come from Tables F, G, I, L. Rows of other "
            "models are added once their runs exist.",
        ]
    )


def main() -> None:
    """Run the report table builder CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(EVAL_SETS_BY_DATASET))
    args = parser.parse_args()

    names = load_dataset_yaml(args.dataset)["names"]
    CLASSES[:] = list(names.values()) if isinstance(names, dict) else names
    b = Builder(args.dataset)
    runs = b.runs_by_model()
    for seeds in runs.values():
        for run_id in seeds.values():
            flag = RUNS_DIR / run_id / "repos.json"
            dirty = (
                sorted(
                    n
                    for n, s in json.loads(flag.read_text()).items()
                    if s["dirty"]
                )
                if flag.exists()
                else []
            )
            if dirty:
                rel_flag = flag.relative_to(REPO_ROOT)
                b.warnings.append(
                    f"{rel_flag}: produced with uncommitted code in "
                    f"{', '.join(dirty)}"
                )
    out_dir = RESULTS_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_A_dataset.md": table_a(b),
        "table_B_training.md": table_b(b, runs),
        "table_C_environment.md": table_c(b, runs),
        "table_D_fp32.md": table_d(b, runs),
        "table_E_complexity.md": table_e(b, runs),
        "table_F_quantization.md": table_f(b, runs),
        "table_G_latency.md": table_g(b, runs),
        "table_H_hardware.md": table_h(b, runs),
        "table_I_deployment.md": table_i(b, runs),
        "table_K_layers.md": table_k(b, runs),
        "table_L_resource.md": table_l(b, runs),
        "table_J_M.md": table_j_m(b),
    }
    for name, text in tables.items():
        (out_dir / name).write_text(
            text + "\n\n_Generated by `make tables`. Do not edit by hand._\n"
        )
    with (out_dir / "tables.csv").open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["table", "row", "column", "value", "source"]
        )
        w.writeheader()
        w.writerows(b.cells)
    problems, warnings = sorted(set(b.problems)), sorted(set(b.warnings))
    n_tbd = sum(1 for c in b.cells if c["value"] == "")
    run_summary = (
        ", ".join(f"{m}: seeds {sorted(s)}" for m, s in runs.items()) or "none"
    )
    sources = (
        [
            f"# Sources: `{args.dataset}`, protocol {PROTOCOL_VERSION}",
            "",
            f"Runs: {sum(len(s) for s in runs.values())} ({run_summary})",
            f"Cells: {len(b.cells)}, missing: {n_tbd}",
            "",
            "## Errors (tables unusable until fixed)",
            "",
        ]
        + ([f"- {p}" for p in problems] or ["- none"])
        + ["", "## Warnings", ""]
        + ([f"- {w}" for w in warnings] or ["- none"])
        + [
            "",
            "## Per-cell sources",
            "",
            "See `tables.csv` (column `source` = file:field).",
        ]
    )
    (out_dir / "SOURCES.md").write_text("\n".join(sources) + "\n")
    print("\n".join(sources[:6]))
    for p in problems:
        print("ERROR:", p)
    for w in warnings:
        print("WARNING:", w)
    print(f"Written: {out_dir}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
