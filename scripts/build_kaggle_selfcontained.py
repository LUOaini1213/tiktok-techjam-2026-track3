#!/usr/bin/env python3
"""
Assemble a SINGLE self-contained Kaggle kernel script from:
  torch_transformer_benchmark.py  (reference harness, minus its __main__ guard)
+ user_optimized.py               (minus the from-benchmark import and __future__)
+ _kaggle_driver.py               (the sweep driver)

No dataset attach needed -> robust headless run. Writes:
  .kaggle_upload/kernel_sc/track3_sc.py
  .kaggle_upload/kernel_sc/kernel-metadata.json
"""

import json
import os
import py_compile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, ".kaggle_upload", "kernel_sc")


def read(p):
    with open(os.path.join(HERE, p), encoding="utf-8") as f:
        return f.read()


def main():
    bench = read("torch_transformer_benchmark.py")
    marker = 'if __name__ == "__main__":'
    assert marker in bench
    bench = bench[: bench.index(marker)].rstrip() + "\n"

    uo = read("user_optimized.py")
    uo = uo.replace("from __future__ import annotations\n", "")
    uo = uo.replace("from torch_transformer_benchmark import BaselineTransformer\n", "")

    driver = read(os.path.join("scripts", "_kaggle_driver.py"))

    combined = (
        bench
        + "\n\n# ================= user_optimized.py (inlined) =================\n\n"
        + uo
        + "\n\n# ================= sweep driver =================\n\n"
        + driver
    )

    os.makedirs(OUTDIR, exist_ok=True)
    out_py = os.path.join(OUTDIR, "track3_sc.py")
    with open(out_py, "w", encoding="utf-8") as f:
        f.write(combined)

    meta = {
        "id": "wenjiluo/track3-bench",
        "title": "track3-bench",
        "code_file": "track3_sc.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(OUTDIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    py_compile.compile(out_py, doraise=True)
    print(f"wrote {out_py} ({len(combined)} chars, {combined.count(chr(10))+1} lines)")
    print("py_compile: OK")


if __name__ == "__main__":
    main()
