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


BOOTSTRAP = '''# --- bootstrap: ensure a torch build compatible with the allocated GPU ---
# Kaggle's API-allocated GPU is often a Tesla P100 (sm_60), which the preinstalled
# torch 2.10+cu128 does NOT support (sm_70+ only). Detect an incompatible build and
# reinstall a P100+T4-compatible torch, then re-exec so the new build is loaded.
import os as _os, sys as _sys, subprocess as _sp
if _os.environ.get("_T3_BOOT") != "1":
    _need = False
    try:
        import torch as _t
        if _t.cuda.is_available():
            _cc = "sm_%d%d" % _t.cuda.get_device_capability()
            _need = _cc not in _t.cuda.get_arch_list()
        else:
            _need = True
    except Exception:
        _need = True
    if _need:
        print("[bootstrap] GPU/torch mismatch -> installing torch 2.5.1+cu121 ...", flush=True)
        _sp.run([_sys.executable, "-m", "pip", "install", "-q",
                 "--index-url", "https://download.pytorch.org/whl/cu121",
                 "torch==2.5.1"], check=False)
        _os.environ["_T3_BOOT"] = "1"
        _os.execv(_sys.executable, [_sys.executable] + _sys.argv)
    _os.environ["_T3_BOOT"] = "1"
'''


def read(p):
    with open(os.path.join(HERE, p), encoding="utf-8") as f:
        return f.read()


def main():
    bench = read("torch_transformer_benchmark.py")
    marker = 'if __name__ == "__main__":'
    assert marker in bench
    bench = bench[: bench.index(marker)].rstrip() + "\n"
    # Inject the GPU-compat bootstrap right after the mandatory __future__ import.
    fut = "from __future__ import annotations\n"
    assert fut in bench
    bench = bench.replace(fut, fut + "\n" + BOOTSTRAP, 1)

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
        "enable_internet": True,
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
