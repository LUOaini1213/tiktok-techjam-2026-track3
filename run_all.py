#!/usr/bin/env python3
"""
Sweep the 14 official test shapes through submission.py in fresh subprocesses
and collect correctness + latency into results/results.csv.

Each shape runs in its own process (matching how the harness is meant to be
used, and avoiding torch.compile cache bleed between shapes).

NOTE on shape 14 (seq_len=100000): the official harness runs the BASELINE first
for the accuracy check, and the baseline materializes a [B,H,S,S] score matrix
(~20.5 TB for shape 14) -> it OOMs and the process dies before the optimized
model ever runs. Shape 14 is therefore handled separately by
``scripts/shape14_optimized_only.py`` (timing + truncated-S correctness).

Usage:
    python run_all.py                       # shapes 1-13 on the auto device
    python run_all.py --shapes 1,5,8,13
    python run_all.py --dtype float16 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Shape:
    idx: int
    batch: int
    d_model: int
    heads: int
    seq: int
    layers: int
    ffn: int
    baseline_feasible: bool = True


# [B, d_model, heads, seq, layers, ffn]  (all causal=True)
SHAPES = [
    Shape(1, 64, 128, 4, 128, 4, 128),
    Shape(2, 1, 128, 4, 128, 4, 128),
    Shape(3, 4, 128, 4, 128, 4, 128),
    Shape(4, 16, 128, 4, 128, 4, 128),
    Shape(5, 128, 128, 4, 128, 4, 128),
    Shape(6, 10000, 128, 4, 128, 4, 128),
    Shape(7, 64, 32, 4, 128, 4, 32),
    Shape(8, 64, 1024, 4, 128, 4, 1024),
    Shape(9, 64, 128, 1, 128, 4, 128),
    Shape(10, 64, 128, 2, 128, 4, 128),
    Shape(11, 64, 128, 16, 128, 4, 128),
    Shape(12, 64, 128, 4, 32, 4, 128),
    Shape(13, 64, 128, 4, 1024, 4, 128),
    Shape(14, 32, 1024, 16, 100000, 2, 1024, baseline_feasible=False),
]

SUMMARY_RE = re.compile(
    r"summary:\s*(PASS|FAIL)\s*\|\s*max_abs=([0-9.eE+-]+)\s*\|\s*max_rel=([0-9.eE+-]+)"
)
BASELINE_RE = re.compile(r"baseline\s*:\s*median=([0-9.eE+-]+)\s*ms")
OPT_RE = re.compile(r"optimized:\s*median=([0-9.eE+-]+)\s*ms")
SPEEDUP_RE = re.compile(r"speedup\s*:\s*([0-9.eE+-]+)x")


def run_shape(sh: Shape, device: str, dtype: str, trials: int, timeout: int) -> dict:
    cmd = [
        sys.executable, os.path.join(HERE, "submission.py"),
        "--causal",
        "--device", device,
        "--dtype", dtype,
        "--batch-size", str(sh.batch),
        "--d-model", str(sh.d_model),
        "--heads", str(sh.heads),
        "--seq-len", str(sh.seq),
        "--layers", str(sh.layers),
        "--ffn-dim", str(sh.ffn),
        "--accuracy-trials", str(trials),
    ]
    row = {
        "shape": sh.idx, "B": sh.batch, "D": sh.d_model, "H": sh.heads,
        "S": sh.seq, "L": sh.layers, "F": sh.ffn,
        "pass": "", "max_abs": "", "max_rel": "",
        "baseline_ms": "", "opt_ms": "", "speedup": "", "notes": "",
    }
    if not sh.baseline_feasible:
        row["notes"] = "baseline OOM (20.5TB scores); see scripts/shape14_optimized_only.py"
        return row

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=HERE
        )
        out = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        row["notes"] = f"timeout>{timeout}s"
        return row

    m = SUMMARY_RE.search(out)
    if m:
        row["pass"], row["max_abs"], row["max_rel"] = m.group(1), m.group(2), m.group(3)
    mb, mo, ms = BASELINE_RE.search(out), OPT_RE.search(out), SPEEDUP_RE.search(out)
    if mb:
        row["baseline_ms"] = mb.group(1)
    if mo:
        row["opt_ms"] = mo.group(1)
    if ms:
        row["speedup"] = ms.group(1)
    if not m and ("out of memory" in out.lower() or "OutOfMemory" in out):
        row["notes"] = "CUDA OOM"
    elif not m:
        row["notes"] = f"no summary (exit {proc.returncode}); check log"
    row["notes"] = (row["notes"] + f" | {time.time()-t0:.0f}s").strip(" |")
    return row


def parse_shapes(spec: str):
    if spec in ("", "all"):
        return list(SHAPES)
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        elif part:
            ids.add(int(part))
    return [s for s in SHAPES if s.idx in ids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1-13", help="e.g. 1-13 | 1,5,8 | all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32",
                    choices=("float32", "float16", "bfloat16"))
    ap.add_argument("--accuracy-trials", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.csv"))
    args = ap.parse_args()

    shapes = parse_shapes(args.shapes)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    speedups = []
    for sh in shapes:
        print(f"\n===== shape {sh.idx}: B={sh.batch} D={sh.d_model} H={sh.heads} "
              f"S={sh.seq} L={sh.layers} F={sh.ffn} =====", flush=True)
        row = run_shape(sh, args.device, args.dtype, args.accuracy_trials, args.timeout)
        print(f"  pass={row['pass']} max_abs={row['max_abs']} max_rel={row['max_rel']} "
              f"speedup={row['speedup']} notes={row['notes']}", flush=True)
        rows.append(row)
        if row["pass"] == "PASS" and row["speedup"]:
            speedups.append(float(row["speedup"]))

    fields = ["shape", "B", "D", "H", "S", "L", "F",
              "pass", "max_abs", "max_rel", "baseline_ms", "opt_ms", "speedup", "notes"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(1 for r in rows if r["pass"] == "PASS")
    n_grade = sum(1 for r in rows if SHAPES[r["shape"] - 1].baseline_feasible)
    print(f"\n=== DONE === PASS {n_pass}/{n_grade} gradeable | wrote {args.out}")
    if speedups:
        speedups.sort()
        mid = speedups[len(speedups) // 2]
        print(f"median speedup across PASSed shapes: {mid:.3f}x "
              f"(min {min(speedups):.3f}x, max {max(speedups):.3f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
