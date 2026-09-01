#!/usr/bin/env python3
"""
Assemble the shareable HTML report from the template plus the measured CSVs, so
the page can never drift from the data it describes.

Substitutes __MEDIAN__ / __RANGE__ from results/results_t4.csv and inlines
report/figures/*.png as data URIs (the Artifact CSP blocks external images).

Usage: python scripts/build_report_page.py <template.html> <output.html>
"""

from __future__ import annotations

import base64
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def speedups(path):
    with open(path, encoding="utf-8") as f:
        return [float(r["speedup"]) for r in csv.DictReader(f)
                if r["pass"] == "PASS" and r["speedup"]]


def data_uri(rel):
    with open(os.path.join(HERE, rel), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    tpl, out = sys.argv[1], sys.argv[2]
    sp = sorted(speedups(os.path.join(HERE, "results", "results_t4.csv")))
    median = sp[len(sp) // 2]

    html = open(tpl, encoding="utf-8").read()
    html = html.replace("__MEDIAN__", f"{median:.3f}")
    html = html.replace("__RANGE__", f"{sp[0]:.3f}–{sp[-1]:.3f}×")
    html = html.replace("__FIG_MEMWALL__", data_uri("report/figures/memory_wall.png"))
    html = html.replace("__FIG_SPEEDUPS__", data_uri("report/figures/speedups.png"))

    assert "__" not in html.replace("__", "", 0) or "__MEDIAN__" not in html
    for token in ("__MEDIAN__", "__RANGE__", "__FIG_MEMWALL__", "__FIG_SPEEDUPS__"):
        assert token not in html, f"unsubstituted placeholder: {token}"

    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"wrote {out} ({len(html)/1024:.0f} KB)")
    print(f"  T4 median {median:.3f}x over {len(sp)} PASS shapes "
          f"(min {sp[0]:.3f}x, max {sp[-1]:.3f}x)")


if __name__ == "__main__":
    main()
