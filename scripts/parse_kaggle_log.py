#!/usr/bin/env python3
"""
Parse a Kaggle kernel log (the JSON event array from `kaggle kernels output`)
produced by the self-contained sweep, and write results/results.csv.

Usage: python scripts/parse_kaggle_log.py <path-to-track3-bench.log>
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHAPES = {
    1: (64, 128, 4, 128, 4, 128), 2: (1, 128, 4, 128, 4, 128),
    3: (4, 128, 4, 128, 4, 128), 4: (16, 128, 4, 128, 4, 128),
    5: (128, 128, 4, 128, 4, 128), 6: (10000, 128, 4, 128, 4, 128),
    7: (64, 32, 4, 128, 4, 32), 8: (64, 1024, 4, 128, 4, 1024),
    9: (64, 128, 1, 128, 4, 128), 10: (64, 128, 2, 128, 4, 128),
    11: (64, 128, 16, 128, 4, 128), 12: (64, 128, 4, 32, 4, 128),
    13: (64, 128, 4, 1024, 4, 128), 14: (32, 1024, 16, 100000, 2, 1024),
}
SUMMARY_RE = re.compile(r"summary:\s*(PASS|FAIL)\s*\|\s*max_abs=([0-9.eE+-]+)\s*\|\s*max_rel=([0-9.eE+-]+)")
BASE_RE = re.compile(r"baseline\s*:\s*median=([0-9.eE+-]+)\s*ms")
OPT_RE = re.compile(r"optimized:\s*median=([0-9.eE+-]+)\s*ms")
SPEED_RE = re.compile(r"speedup\s*:\s*([0-9.eE+-]+)x")


def load_stdout(path):
    """Accept either the JSON event array from `kaggle kernels output` or a
    plain-text log (e.g. copied from the Kaggle web console)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        events = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return "".join(e["data"] for e in events if e.get("stream_name") == "stdout")


def _round(entry, sig=6):
    """Trim float reprs (1.1920928955078125e-06) to a readable precision."""
    out = dict(entry)
    for key in ("max_abs", "max_rel"):
        try:
            out[key] = f"{float(out[key]):.{sig}g}"
        except (TypeError, ValueError):
            pass
    return out


def parse_summary_block(text):
    """The driver ends with a compact CSV block; prefer it when present.

    Format: shape,pass,max_abs,max_rel,baseline_ms,opt_ms,speedup,note
    It is authoritative because the driver writes it from the values it
    measured, rather than us re-scraping prose.
    """
    head = re.escape("SUMMARY (copy from here)")
    tail = re.escape("END SUMMARY")
    m = re.search(head + "(.*?)" + tail, text, re.S)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        cells = line.split(",")
        if len(cells) < 7 or not cells[0].strip().isdigit():
            continue
        idx = int(cells[0])
        note = ",".join(cells[7:]).strip() if len(cells) > 7 else ""
        out[idx] = {
            "pass": cells[1].strip(), "max_abs": cells[2].strip(),
            "max_rel": cells[3].strip(), "baseline_ms": cells[4].strip(),
            "opt_ms": cells[5].strip(), "speedup": cells[6].strip(),
            "notes": note,
        }
    return out or None


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "kaggle.log")
    text = load_stdout(log)
    summary = parse_summary_block(text)
    env = ""
    m = re.search(r"gpu .*", text)
    if m:
        env = m.group(0).strip()
    # split into per-shape blocks
    parts = re.split(r"##### SHAPE (\d+)[^\n]*#####", text)
    blocks = {}
    for i in range(1, len(parts), 2):
        blocks[int(parts[i])] = parts[i + 1]

    rows = []
    speedups = []
    for idx in range(1, 14):
        blk = blocks.get(idx, "")
        b, d, h, s, l, f = SHAPES[idx]
        row = {"shape": idx, "B": b, "D": d, "H": h, "S": s, "L": l, "F": f,
               "pass": "", "max_abs": "", "max_rel": "", "baseline_ms": "",
               "opt_ms": "", "speedup": "", "notes": ""}
        if summary and idx in summary:
            row.update(_round(summary[idx]))
        else:
            sm = SUMMARY_RE.search(blk)
            if sm:
                row["pass"], row["max_abs"], row["max_rel"] = sm.group(1), sm.group(2), sm.group(3)
            for rx, key in ((BASE_RE, "baseline_ms"), (OPT_RE, "opt_ms"), (SPEED_RE, "speedup")):
                mm = rx.search(blk)
                if mm:
                    row[key] = mm.group(1)
        # Only scrape an error out of the prose when no verdict was recovered
        # above -- and test that, not a regex match that the summary branch
        # never binds.
        if not row["pass"] and "RuntimeError" in blk:
            em = re.search(r"(RuntimeError|OutOfMemoryError)[^\n]*", blk)
            row["notes"] = (em.group(0)[:80] if em else "error")
        if row["pass"] == "PASS" and row["speedup"]:
            speedups.append(float(row["speedup"]))
        rows.append(row)

    # shape 14 block
    blk14 = blocks.get(14, "")
    row14 = {"shape": 14, "B": 32, "D": 1024, "H": 16, "S": 100000, "L": 2, "F": 1024,
             "pass": "", "max_abs": "", "max_rel": "", "baseline_ms": "",
             "opt_ms": "", "speedup": "", "notes": "baseline infeasible ~20.5TB"}
    tc = re.search(r"trunc S=\d+ correctness:\s*(PASS|FAIL)[^\n]*", blk14)
    full = re.search(r"full S=100000:[^\n]*", blk14)
    if tc:
        row14["pass"] = tc.group(1) + "(trunc)"
        mr = re.search(r"max_rel=([0-9.eE+-]+)", tc.group(0)); ma = re.search(r"max_abs=([0-9.eE+-]+)", tc.group(0))
        if mr: row14["max_rel"] = mr.group(1)
        if ma: row14["max_abs"] = ma.group(1)
    if summary and 14 in summary:
        s14 = _round(summary[14])
        for k in ("pass", "max_abs", "max_rel"):
            if s14[k]:
                row14[k] = s14[k]
        if summary[14]["notes"]:
            row14["notes"] = summary[14]["notes"]
    if full:
        row14["notes"] = full.group(0).strip()[:120]
    elif re.search(r"shape14 full-seq", blk14):
        em = re.search(r"(OutOfMemory|RuntimeError)[^\n]*", blk14)
        row14["notes"] = "full-seq: " + (em.group(0)[:90] if em else "error")
    rows.append(row14)

    out = os.path.join(HERE, "results", "results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ["shape", "B", "D", "H", "S", "L", "F", "pass", "max_abs",
              "max_rel", "baseline_ms", "opt_ms", "speedup", "notes"]
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print("ENV:", env)
    print(f"{'sh':>2} {'pass':<6} {'max_abs':>9} {'max_rel':>9} {'base_ms':>9} {'opt_ms':>9} {'speedup':>7}  notes")
    for r in rows:
        print(f"{r['shape']:>2} {r['pass']:<6} {r['max_abs']:>9} {r['max_rel']:>9} "
              f"{r['baseline_ms']:>9} {r['opt_ms']:>9} {r['speedup']:>7}  {r['notes']}")
    n_pass = sum(1 for r in rows[:13] if r["pass"] == "PASS")
    print(f"\nPASS {n_pass}/13 gradeable shapes")
    if speedups:
        speedups.sort()
        print(f"speedup: median={speedups[len(speedups)//2]:.3f}x "
              f"min={min(speedups):.3f}x max={max(speedups):.3f}x")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
