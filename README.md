# Track 3 — Implement a GPU Kernel for a Transformer Layer
### TikTok TechJam 2026 · "Build with Joy, Code for Change"

Optimize the runtime of a Transformer forward pass on a GPU while keeping the
output numerically identical to the reference implementation (per-element
`abs_err ≤ 0.002` **OR** `rel_err ≤ 0.02`), across 14 official test shapes.

## TL;DR — what we do

| Lever | Effect |
|---|---|
| **`F.scaled_dot_product_attention` (FlashAttention)** | Replaces the baseline's `O(S²)` materialized score matrix with an `O(S)` fused kernel. Makes the `seq_len=100000` shape possible at all — the baseline would need **~20.5 TB** just for its attention scores. Measured: the full 100k-token forward completes in **14.6 GB** on a free P100. |
| **Internal fp16 autocast** (`T3_AUTOCAST=fp16`, **opt-in**) | Lights up the tensor cores and roughly **doubles** the median speedup (2.36x -> 4.01x). Shipped **off**: it passes all 13 shapes, but its worst absolute error has already crossed `atol=0.002` and survives on the relative branch alone. Reductions stay in fp32. |
| **Self-applied `torch.compile`** | Fuses LayerNorm / bias / GELU epilogues into Triton kernels; independent of the grader passing `--compile-user`. Needs sm≥7.0, so it is inactive on a P100. Measured on a T4 it is worth +0.3x on compute-bound shapes and is a **net loss** on the launch-bound one — see the ablation. |
| **Batch chunking into a preallocated output** (only `seq_len=1e5`) | Keeps activations inside a 16 GB GPU. Chunks are written in place rather than collected and `torch.cat`-ed — the concat holds the pieces *and* the joined result at once, which is what made this shape OOM. |

We keep **every baseline submodule and parameter name unchanged** and rewrite
only the forward compute, so the harness' `copy_model_weights(strict=True)`
succeeds and the comparison is apples-to-apples.

## Files

```
torch_transformer_benchmark.py   official reference & harness (UNMODIFIED)
tensorflow_transformer_benchmark.py  the TF half of the official problem
                                 statement, shipped UNMODIFIED for reference;
                                 this submission is the PyTorch track only
user_optimized.py                UserOptimizedTransformer — the deliverable
submission.py                    runs the official main() with our model swapped in
run_all.py                       sweeps shapes 1–13 -> results/results.csv
scripts/shape14_optimized_only.py  seq_len=1e5 timing + truncated-S correctness
notebooks/colab_run.ipynb        shapes 1–13 on a free Colab T4
notebooks/kaggle_shape14.ipynb   shape 14 on Kaggle T4/P100
results/results.csv             per-shape PASS / max_abs / speedup
results/kaggle_p100_run.log     raw log of the run those numbers come from
results/ablation.md  report/  docs/AI_TOOLS.md
```

## Setup

GPU work runs on **free cloud GPUs** (Google Colab T4 / Kaggle T4 or P100);
`torch`/`triton` are preinstalled there. One caveat we hit: Kaggle's *default*
API-allocated card is a Tesla P100 (`sm_60`), which the preinstalled
torch 2.10+cu128 no longer supports (`arch_list` starts at `sm_70`), so the
kernel builder detects the mismatch and installs torch 2.5.1+cu121 before
re-exec'ing. Ask for a T4 instead and no reinstall happens — pass
`--accelerator NvidiaTeslaT4` to `kaggle kernels push`.
No local GPU is required (this project was developed on a machine with only an
Intel iGPU; the tech report states the exact cloud environment used).

```bash
# On a Colab/Kaggle GPU runtime, from the repo root:
python submission.py --causal --device cuda --dtype float32 \
  --batch-size 64 --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128
```

## Reproduce the results

There are two paths, and it matters which one produced the committed numbers.

**What the committed `results/results.csv` actually came from.** A single
self-contained Kaggle kernel, built and pushed from this machine, because the
laptop has no NVIDIA GPU:

```bash
python scripts/build_kaggle_selfcontained.py --accelerator NvidiaTeslaT4
kaggle kernels push -p .kaggle_upload/kernel_sc      # runs shapes 1-13, ablation, 14
kaggle kernels output wenjiluo/track3-bench -p .kaggle_out
python scripts/parse_kaggle_log.py .kaggle_out/track3-bench.log   # -> results/results.csv
```

The builder inlines the unmodified harness, `user_optimized.py` and the sweep
driver into one script, so the kernel needs no dataset attach. Kaggle's API
hands out a **P100 by default**; `--accelerator NvidiaTeslaT4` asks for a T4
instead (the accepted names are `NvidiaTeslaT4`, `NvidiaTeslaP100`,
`Tpu1VmV38` -- anything else is silently normalised back to a P100).

**If you already have a GPU**, the same model runs under the official harness
directly:

```bash
python run_all.py --shapes 1-13 --device cuda --dtype float32 --out /tmp/mine.csv
python scripts/shape14_optimized_only.py --trunc-seq 2048
```

`run_all.py` runs each shape in a fresh subprocess and parses the harness' own
`summary: PASS/FAIL`, `max_abs`, `max_rel` and median speedup. It is the more
faithful check -- it is the official `main()` -- but it is not what produced the
table below, and we would rather say so than let the two look interchangeable.

## Results

Two free Kaggle GPUs, both under fp32 grading, both **13/13 gradeable shapes
PASS**. Raw logs are committed next to every table so each number can be traced
to the run that produced it.

| GPU | what it can run | median speedup | range | worst `max_abs` |
|---|---|---|---|---|
| Tesla P100 (`sm_60`) | SDPA only | 2.065x | 1.098x - 4.001x | 1.91e-6 |
| **Tesla T4 (`sm_75`)** | **SDPA + `torch.compile` + FlashAttention** | **2.357x** | 1.088x - 4.505x | 1.91e-6 |
| Tesla T4, `T3_AUTOCAST=fp16` | + fp16 tensor cores | 4.014x | 1.320x - 11.528x | 2.04e-3 — *see below* |

Kaggle's API hands out a P100 unless you ask otherwise, which is why the first
row exists: on `sm_60` Triton will not build, so `torch.compile` never engages
and SDPA falls back to its memory-efficient backend. Pass
`--accelerator NvidiaTeslaT4` and the full stack lights up.

Per shape, fp32 (`results/results.csv` = P100, `results/results_t4.csv` = T4):

| # | B,D,H,S,F | P100 base -> opt | P100 | T4 base -> opt | T4 |
|---|---|---|---|---|---|
| 1 | 64,128,4,128,128 | 5.91 -> 3.37 | 1.76x | 9.47 -> 4.02 | 2.36x |
| 2 | 1,128,4,128,128 | 3.16 -> 1.48 | 2.14x | 3.03 -> 0.89 | 3.40x |
| 3 | 4,128,4,128,128 | 3.18 -> 1.46 | 2.17x | 3.02 -> 1.02 | 2.95x |
| 4 | 16,128,4,128,128 | 3.15 -> 1.38 | 2.29x | 2.99 -> 1.20 | 2.49x |
| 5 | 128,128,4,128,128 | 11.00 -> 6.19 | 1.78x | 18.28 -> 9.25 | 1.98x |
| 6 | 10000,128,4,128,128 | 772.3 -> 418.0 | 1.85x | 1431.8 -> 750.1 | 1.91x |
| 7 | 64,32,4,128,32 | 4.05 -> 1.96 | 2.07x | 6.28 -> 2.31 | 2.72x |
| 8 | 64,1024,4,128,1024 | 70.9 -> 64.6 | 1.10x | 127.4 -> 117.1 | 1.09x |
| 9 | 64,128,1,128,128 | 3.85 -> 3.11 | 1.24x | 6.36 -> 4.97 | 1.28x |
| 10 | 64,128,2,128,128 | 4.77 -> 3.13 | 1.52x | 7.88 -> 4.94 | 1.60x |
| 11 | 64,128,16,128,128 | 12.55 -> 4.91 | 2.56x | 22.31 -> 7.07 | 3.16x |
| 12 | 64,128,4,32,128 | 3.06 -> 1.32 | 2.33x | 2.97 -> 1.52 | 1.95x |
| 13 | 64,128,4,1024,128 | 168.6 -> 42.1 | **4.00x** | 316.8 -> 70.3 | **4.51x** |

One honest observation from that table: **the T4's baselines are slower than the
P100's** (shape 13: 316.8 ms vs 168.6 ms). The P100 has more fp32 throughput and
about twice the memory bandwidth. The T4 ratios are better anyway because our
path picks up compile and FlashAttention there while the baseline stays
bandwidth-bound. A speedup is a ratio; it is worth saying which side moved.

### fp16 is twice as fast, and we still ship fp32

`T3_AUTOCAST=fp16` passes all 13 shapes at a **median 4.014x**, up to 11.53x on
shape 13. We ship it **off**. The reason is one number: its worst-case absolute
error is `max_abs = 0.0020388` on shape 6, which has **already crossed the
`atol=0.002` gate**. It passes only because the rule is
`abs<=0.002` **OR** `rel<=0.02` and that element's reference value happened to be
large enough (`|ref| >= 0.102`) for the relative branch to catch it.

Nothing makes that repeatable. Put the same error on a near-zero reference and
the element fails, and one failing element fails the shape and forfeits the
speed score entirely. The shipped fp32 path sits **1049x inside** the same gate.
We took the margin. The flag is documented, measured, and yours if you want the
other side of the trade — full numbers in [`results/ablation.md`](results/ablation.md).

### Shape 14: the one the baseline cannot run

Its score matrix is `[32, 16, 100000, 100000]` = 5.12e12 elements = **~20.5 TB**
in fp32. No GPU holds that, so there is no baseline time to divide by - the
result is not a speedup number, it is the difference between *cannot run* and
*runs*. On the same free 16 GB P100:

```
trunc S=2048 correctness: PASS max_abs=1.19e-06 max_rel=0.127
vram free=16.64/17.06 GB | baseline scores would be 20.5 TB -> infeasible
full S=100000: median=293376.9 ms | 10,907 tok/s | peak_vram=14.61 GB | chunk_bs=1
```

293 s per forward over 3.2 M tokens, peak 14.6 GB. Correctness is validated at a
truncated `seq_len` where the baseline *can* run (PASS, `max_abs 1.2e-6`); SDPA's
math does not depend on `S`, so passing there evidences correctness at 1e5.

**Precision, stated plainly:** the truncated correctness check is **fp32**, the
same dtype as every graded shape. The full-length *timing* is **fp16**, and that
is forced, not chosen: an fp32 input for this shape is
`32 x 100000 x 1024 x 4 B = 13.1 GB`, and the output is another 13.1 GB — 26.2 GB
of unavoidable tensors before a single activation, which no free 16 GB GPU can
hold. In fp16 the pair is 13.1 GB total and it fits. The 293 s / 10,907 tok/s /
14.6 GB figures above are therefore fp16 figures; the 13 graded shapes are not.
Note this is a **native fp16 run** (the driver casts the whole model and input),
not the `T3_AUTOCAST` knob — that knob deliberately disables itself whenever the
incoming dtype is not fp32.

Two caveats we would rather state than hide: the P100 is `sm_60`, so it gets
neither the real FlashAttention kernel (SDPA falls back to the memory-efficient
backend) nor `torch.compile` (Triton needs sm>=7.0). Every speedup above is
therefore **from SDPA alone**; a T4/A100 with compilation enabled would be
higher. See `report/figures/`.

## Ablation

Every stage is selected by environment variable with **no code edits**, so the
ablation and the delivered path are literally the same code. The measured
4-shape x 4-stage table is in [`results/ablation.md`](results/ablation.md)
(machine-readable: `results/ablation_t4.csv`).

Shipped defaults are `T3_AUTOCAST=off`, `T3_COMPILE=1`, i.e. **SDPA + compile**
(compile activates only on `sm>=7.0`). Pass `--out` so a stage run does not
overwrite the committed `results/results.csv`:

```bash
T3_COMPILE=0 T3_AUTOCAST=off  python run_all.py --shapes 1 --out /tmp/sdpa.csv
T3_COMPILE=1 T3_AUTOCAST=off  python run_all.py --shapes 1 --out /tmp/compile.csv
T3_COMPILE=0 T3_AUTOCAST=fp16 python run_all.py --shapes 1 --out /tmp/fp16.csv
T3_COMPILE=1 T3_AUTOCAST=fp16 python run_all.py --shapes 1 --out /tmp/both.csv
```

Environment toggles: `T3_AUTOCAST` (auto|fp16|bf16|off), `T3_COMPILE` (1|0),
`T3_COMPILE_MODE`, `T3_FP32_FFN` (1|0), `T3_CHUNK_BS`.

## Correctness notes

- Every element must pass (zero failures); `NaN/Inf` is a hard fail. Measured
  worst-case `max_abs = 1.91e-6` across all 13 shapes, ~1050× inside `atol=0.002`.
- The `max_rel` column looks large (up to ~15) and that is expected: the harness
  computes it over *every* element as `abs_err / max(|ref|, 1e-12)`, so an output
  whose reference is ~1e-7 reports a huge ratio while its absolute error is still
  ~1e-6. Those elements pass on the absolute criterion — which is exactly what the
  `abs OR rel` rule exists for. **Zero elements fail either test.**
- Softmax accumulates in fp32 (SDPA on CUDA), matching the reference; GELU uses
  the exact `approximate="none"` form.
- Shape 14 has no runnable baseline (20.5 TB scores). We validate correctness at
  a truncated `seq_len` where the baseline runs, and argue by construction that
  SDPA is shape-invariant.

## Limitations & future work

- fp16 is measured across all 13 shapes, not assumed: it passes, at 4.01x
  median, with `max_abs` 2.04e-3 against a 2e-3 gate. A precision ladder
  (`T3_FP32_FFN`, fp32 SDPA) exists for anyone who enables it and needs to claw
  margin back on a specific shape. What we have *not* done is find a mixed
  assignment -- fp16 matmuls with selected fp32 stages -- that keeps most of the
  2x while restoring real margin. That is the obvious next experiment.
- The stage ablation covers 4 representative shapes on one GPU, one run each;
  the 13-shape sweeps are the ones with repeat trials behind them.
- Hand-written Triton kernels (fused LayerNorm+residual, fused bias+GELU) were
  scoped but **not built** — there is no `kernels/` directory. The core path is
  SDPA + `torch.compile`, and Inductor already fuses those epilogues.
- A hand-written Turing FlashAttention kernel is out of scope (multi-day effort).
- **`--dtype bfloat16` fails, and not because of us.** The harness accepts it
  (`run_all.py --dtype bfloat16`), and we do fail it: 6131/131072 elements,
  `max_abs 0.047`. But so does the *reference compared against itself* recomputed
  in fp32 — 7603/131072 elements at the identical `max_abs 0.047`. bf16's ulp
  near 1.0 is 0.0078, four times coarser than `atol=0.002`, so no implementation
  that reorders a single operation can hold that gate in bf16. The graded
  configuration is fp32 (the harness default), where we sit ~1049x inside it.
- The padded (`padding_ratio > 0`) fallback still materializes a dense
  `[B,1,S,S]` additive bias, i.e. it gives back the `O(S^2)` memory that SDPA
  exists to avoid. The graded path runs `padding_ratio=0`, where masking is
  kernel-generated via `is_causal` and no bias is built; the fallback only has to
  be correct at the small `S` where a mask is actually supplied. Making the
  padded path memory-efficient too (block-sparse or a folded key-padding mask)
  is unfinished work, not a solved problem we left out.

## AI tooling

Design and implementation were driven with Claude (Claude Code). See
[`docs/AI_TOOLS.md`](docs/AI_TOOLS.md) for the prompt → design → diff → verify log.
