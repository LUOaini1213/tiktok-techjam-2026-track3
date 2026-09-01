# Ablation study

Every stage is selected by an **environment variable, with no code edits**, so
the numbers here and the delivered path are literally the same code:

| Knob | Values | Default |
|---|---|---|
| `T3_COMPILE` | `1` / `0` | `1` (activates only on `sm>=7.0`) |
| `T3_AUTOCAST` | `off` / `fp16` / `bf16` / `auto` | `off` |
| `T3_FP32_FFN` | `1` / `0` | `0` |
| `T3_CHUNK_BS` | int | auto, from free VRAM |

```bash
T3_COMPILE=0 T3_AUTOCAST=off  python run_all.py --shapes 1 --out /tmp/sdpa.csv
T3_COMPILE=1 T3_AUTOCAST=off  python run_all.py --shapes 1 --out /tmp/compile.csv
T3_COMPILE=0 T3_AUTOCAST=fp16 python run_all.py --shapes 1 --out /tmp/fp16.csv
T3_COMPILE=1 T3_AUTOCAST=fp16 python run_all.py --shapes 1 --out /tmp/both.csv
```

## Stage table (Kaggle Tesla T4, sm_75, fp32 grading)

4 shapes x 4 stages, one run each. Raw log: `kaggle_t4_ablation.log`; machine
readable: `ablation_t4.csv`.

| shape | what it stresses | baseline ms | SDPA | +compile | +fp16 | +compile+fp16 |
|---|---|---|---|---|---|---|
| 1 (B=64,S=128) | compute | 9.4 | 2.01x | 2.33x | 3.71x | **5.61x** |
| 12 (S=32) | launch overhead | 3.4 | 2.29x | 1.46x | 1.42x | **2.86x** |
| 8 (D=ffn=1024) | GEMM / wide FFN | 118.9 | 1.07x | 1.05x | **3.80x** | 3.80x |
| 13 (S=1024) | long sequence | 313.6 | 4.68x | 4.63x | **11.57x** | 11.54x |

What each stage is actually worth:

- **SDPA is the whole story on long sequences.** Shape 13 gets 4.68x from
  memory-efficient attention alone, and nothing later adds to that except fp16.
- **`torch.compile` is not free.** It helps the compute-bound shape (2.01 ->
  2.33x) and it *hurts* the launch-bound one (2.29 -> 1.46x): shape 12 is
  3.4 ms of almost pure kernel-launch overhead, and Inductor's guard and
  dispatch overhead is a real fraction of that. On the GEMM-bound shape 8 it is
  a wash (1.07 -> 1.05x) because cuBLAS already owns the time.
- **fp16 is the biggest single lever** — and the one we do not ship. See below.
- **The stages are not additive.** On shape 8 and 13, compile contributes
  nothing once fp16 is on; on shape 12 the two only pay off together.

## The fp16 decision, with the numbers

fp16 is **off by default** (`T3_AUTOCAST=off`). The repo previously justified
that by asserting fp16 "breaks the strict `atol=0.002` gate". **That assertion
was wrong, and measuring it is what showed us why the conclusion is still
right.**

A full 13-shape fp16 sweep on the T4 (`results_t4_fp16.csv`,
`kaggle_t4_fp16_run.log`) **passes all 13 shapes** at a **median 4.014x**
(1.320x - 11.528x), against 2.282x for the shipped fp32 path. That is nearly
double, for one environment variable.

Here is why we still do not ship it:

| regime | shapes PASS | median speedup | worst `max_abs` | margin vs `atol=0.002` |
|---|---|---|---|---|
| fp32 (shipped) | 13/13 | 2.282x | 1.91e-6 | **1049x** |
| fp16 (`T3_AUTOCAST=fp16`) | 13/13 | 4.014x | **2.04e-3** | **0.98x** |

The fp16 worst case, shape 6, is `max_abs = 0.0020388` — it has **already
crossed `atol=0.002`**. It passes only because the harness gate is
`abs<=0.002` **OR** `rel<=0.02`, and that particular element happened to have a
reference value large enough (`|ref| >= 0.102`) for the relative branch to catch
it. Nothing about the run guarantees the next seed puts that error on an element
with a large reference; if it lands on a near-zero one instead, that element
fails, and a single failing element fails the whole shape and forfeits the speed
score entirely.

So the trade is: **2x more speed, in exchange for a correctness margin of
essentially zero on a hard gate.** We take the 2.282x that sits 1049x inside
tolerance. The flag is there, documented and measured, for anyone who wants the
other side of that trade.

## The chunking stage (shape 14)

The only stage with a binary outcome rather than a ratio.

| variant | outcome |
|---|---|
| baseline (explicit `[B,H,S,S]` scores) | cannot run — needs ~20.5 TB |
| SDPA + chunks collected into a list + `torch.cat` | **OOM** at the concat: `Tried to allocate 6.10 GiB` |
| SDPA + chunks written into a preallocated output | **runs**: 293377 ms, 10,907 tok/s, peak 14.61 GB, `chunk_bs=1` (P100) |

The concat was the whole difference. Collecting 32 chunks and joining them holds
the pieces and the `[32,100000,1024]` result at the same time — a second ~6.5 GB
allocation on a card that already had the 6.5 GB input resident.

All three rows are **fp16**: an fp32 input plus output for this shape is 26.2 GB,
which does not fit a free 16 GB GPU at all, so fp16 is the only regime in which
the comparison exists. The truncated correctness check that backs this shape is
fp32, like every graded shape.

## Cross-GPU: what the hardware is worth

The same code on two free cards, both fp32, both 13/13 PASS:

| | Tesla P100 (sm_60) | Tesla T4 (sm_75) |
|---|---|---|
| `torch.compile` | unavailable (Triton needs sm>=7.0) | active |
| SDPA backend | memory-efficient | FlashAttention |
| median speedup | 2.065x | **2.282x** |
| range | 1.098x - 4.001x | 1.086x - 4.439x |

Note the T4's *baselines* are slower than the P100's (shape 13: 318.7 ms vs
168.6 ms) — the P100 has higher fp32 throughput and roughly twice the memory
bandwidth. The ratio improves on the T4 anyway, because our path picks up
compile and FlashAttention there while the baseline stays bandwidth-bound. The
speedup is a ratio, and it is worth saying which way each side of it moved.
