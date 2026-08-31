#!/usr/bin/env python3
"""
Generate report figures from results/results.csv:
  - figures/speedups.png   : optimized-vs-baseline median speedup per shape (1-13)
  - figures/memory_wall.png: baseline attention-score memory per shape (log scale),
                             showing why shape 14 (~20.5 TB) is infeasible.

Usage: python scripts/make_figures.py
Requires matplotlib (preinstalled on Colab/Kaggle; `pip install matplotlib` locally).
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "results.csv")
FIG = os.path.join(HERE, "report", "figures")

# [B, d_model, heads, seq, layers, ffn] for the memory-wall chart.
SHAPES = {
    1: (64, 128, 4, 128), 2: (1, 128, 4, 128), 3: (4, 128, 4, 128),
    4: (16, 128, 4, 128), 5: (128, 128, 4, 128), 6: (10000, 128, 4, 128),
    7: (64, 32, 4, 128), 8: (64, 1024, 4, 128), 9: (64, 128, 1, 128),
    10: (64, 128, 2, 128), 11: (64, 128, 16, 128), 12: (64, 128, 4, 32),
    13: (64, 128, 4, 1024), 14: (32, 1024, 16, 100000),
}


def speedup_chart():
    ids, sp = [], []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["pass"] == "PASS" and r["speedup"]:
                ids.append(int(r["shape"]))
                sp.append(float(r["speedup"]))
    if not ids:
        print("no PASS speedups yet; skipping speedups.png")
        return
    plt.figure(figsize=(9, 4))
    bars = plt.bar([str(i) for i in ids], sp, color="#3b82f6")
    plt.axhline(1.0, color="#888", ls="--", lw=1)
    for b, v in zip(bars, sp):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}x",
                 ha="center", va="bottom", fontsize=8)
    plt.ylabel("median speedup vs baseline")
    plt.xlabel("shape #")
    plt.title("Optimized Transformer speedup (Kaggle GPU)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "speedups.png"), dpi=130)
    print("wrote speedups.png")


def memory_wall_chart():
    ids = sorted(SHAPES)
    gb = []
    for i in ids:
        b, d, h, s = SHAPES[i]
        elems = b * h * s * s  # [B, H, S, S] score matrix
        gb.append(elems * 4 / 1e9)  # fp32 bytes -> GB
    plt.figure(figsize=(9, 4))
    colors = ["#ef4444" if i == 14 else "#3b82f6" for i in ids]
    plt.bar([str(i) for i in ids], gb, color=colors)
    plt.axhline(16, color="#16a34a", ls="--", lw=1.2, label="16 GB GPU (T4/P100)")
    plt.yscale("log")
    plt.ylabel("baseline attention-score memory (GB, log)")
    plt.xlabel("shape #")
    plt.title("The memory wall: baseline O(S²) scores. Shape 14 ≈ 20,500 GB.")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "memory_wall.png"), dpi=130)
    print("wrote memory_wall.png (shape 14 = %.0f GB)" % gb[ids.index(14)])


def main():
    os.makedirs(FIG, exist_ok=True)
    memory_wall_chart()   # does not need results.csv
    if os.path.exists(CSV):
        speedup_chart()
    else:
        print("results/results.csv not found yet; run run_all.py first for speedups.png")


if __name__ == "__main__":
    main()
