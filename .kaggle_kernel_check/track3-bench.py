#!/usr/bin/env python3
"""
Kaggle kernel entry point. Driven headlessly via the Kaggle API from a local
machine that has no GPU. Copies our code (attached as a Kaggle dataset) into the
working dir, then runs the official-harness sweep on the Kaggle GPU and the
shape-14 demo. All outputs land in /kaggle/working and are pulled back locally.
"""

import glob
import os
import shutil
import sys

WORK = "/kaggle/working"

# Locate the attached code dataset (the folder containing user_optimized.py).
code_dir = None
for s in glob.glob("/kaggle/input/*"):
    if os.path.exists(os.path.join(s, "user_optimized.py")):
        code_dir = s
        break
assert code_dir, f"code dataset not found under /kaggle/input: {glob.glob('/kaggle/input/*')}"

for root, _, files in os.walk(code_dir):
    rel = os.path.relpath(root, code_dir)
    dst = WORK if rel == "." else os.path.join(WORK, rel)
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.copy2(os.path.join(root, f), os.path.join(dst, f))

os.chdir(WORK)
print("working files:", sorted(os.listdir(".")), flush=True)

import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"device {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU!!'} | "
      f"bf16 {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}",
      flush=True)
assert torch.cuda.is_available(), "GPU not enabled on this kernel (phone-verify the account)."

print("\n########## shapes 1-13 (official harness, fp32 grading) ##########", flush=True)
rc = os.system(f"{sys.executable} run_all.py --shapes 1-13 --device cuda --dtype float32")
print("run_all exit:", rc, flush=True)

print("\n########## shape 14 (optimized-only timing + truncated correctness) ##########",
      flush=True)
os.system(f"{sys.executable} shape14_optimized_only.py --trunc-seq 2048")

print("\n########## results/results.csv ##########", flush=True)
try:
    with open("results/results.csv") as f:
        print(f.read(), flush=True)
except FileNotFoundError:
    print("no results.csv produced", flush=True)
