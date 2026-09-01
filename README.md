# Track 3 — Implement a GPU Kernel for a Transformer Layer
### TikTok TechJam 2026 · "Build with Joy, Code for Change"

Optimize the runtime of a Transformer forward pass on a GPU while keeping the
output numerically identical to the reference implementation (per-element
`abs_err ≤ 0.002` **OR** `rel_err ≤ 0.02`), across 14 official test shapes.

## TL;DR — what we do

| Lever | Effect |
|---|---|
| **`F.scaled_dot_product_attention` (FlashAttention)** | Replaces the baseline's `O(S²)` materialized score matrix with an `O(S)` fused kernel. Makes the `seq_len=100000` shape possible at all — the baseline would need **~20.5 TB** just for its attention scores. Measured: the full 100k-token forward completes in **14.6 GB** on a free P100. |
| **Internal fp16 autocast** (`T3_AUTOCAST=fp16`, **opt-in**) | Lights up the tensor cores. Off by default: across 4 layers, fp16 rounding pushes near-zero outputs past the *absolute* `atol=0.002` gate, and correctness outranks speed. Reductions stay in fp32. |
| **Self-applied `torch.compile`** | Fuses LayerNorm / bias / GELU epilogues into Triton kernels; independent of the grader passing `--compile-user`. Per-shape mode (CUDA graphs for launch-bound shapes). Needs sm≥7.0, so it is *inactive* on the P100 the numbers below come from. |
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

Measured on a **free Kaggle Tesla P100** (16 GB, fp32 grading). Full data in
[`results/results.csv`](results/results.csv); the raw kernel log is committed as
[`results/kaggle_p100_run.log`](results/kaggle_p100_run.log).

**13/13 gradeable shapes PASS - median speedup 2.07x - up to 4.00x - max_abs ~1e-6 -
and shape 14, which the baseline cannot run at all, completes at its full
`seq_len=100000` inside 14.6 GB.**

| # | B,D,H,S,F | pass | max_abs | baseline ms | opt ms | speedup |
|---|---|---|---|---|---|---|
| 1 | 64,128,4,128,128 | PASS | 1.2e-6 | 5.91 | 3.37 | 1.76x |
| 2 | 1,128,4,128,128 | PASS | 9.5e-7 | 3.16 | 1.48 | 2.14x |
| 3 | 4,128,4,128,128 | PASS | 9.5e-7 | 3.18 | 1.46 | 2.17x |
| 4 | 16,128,4,128,128 | PASS | 1.1e-6 | 3.15 | 1.38 | 2.29x |
| 5 | 128,128,4,128,128 | PASS | 1.4e-6 | 11.00 | 6.19 | 1.78x |
| 6 | 10000,128,4,128,128 | PASS | 1.9e-6 | 772.3 | 418.0 | 1.85x |
| 7 | 64,32,4,128,32 | PASS | 9.5e-7 | 4.05 | 1.96 | 2.07x |
| 8 | 64,1024,4,128,1024 | PASS | 1.9e-6 | 70.9 | 64.6 | 1.10x |
| 9 | 64,128,1,128,128 | PASS | 1.2e-6 | 3.85 | 3.11 | 1.24x |
| 10 | 64,128,2,128,128 | PASS | 1.2e-6 | 4.77 | 3.13 | 1.52x |
| 11 | 64,128,16,128,128 | PASS | 1.4e-6 | 12.55 | 4.91 | 2.56x |
| 12 | 64,128,4,32,128 | PASS | 1.2e-6 | 3.06 | 1.32 | 2.33x |
| 13 | 64,128,4,1024,128 | PASS | 1.4e-6 | 168.6 | 42.1 | **4.00x** |
| 14 | 32,1024,16,100000,1024 | PASS (see below) | 1.2e-6 | *infeasible (~20.5 TB)* | 293377 | n/a |

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

Two caveats we would rather state than hide: the P100 is `sm_60`, so it gets
neither the real FlashAttention kernel (SDPA falls back to the memory-efficient
backend) nor `torch.compile` (Triton needs sm>=7.0). Every speedup above is
therefore **from SDPA alone**; a T4/A100 with compilation enabled would be
higher. See `report/figures/`.

## Ablation

Every stage is selected by environment variable with **no code edits**, so the
ablation and the delivered path are literally the same code — see
`results/ablation.md` for the measured table.

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

- fp16 tolerance is checked empirically per shape; a precision ladder
  (`T3_FP32_FFN`, fp32-SDPA) is available for any shape that fails.
- Hand-written Triton kernels (fused LayerNorm+residual, fused bias+GELU) were
  scoped but **not built** — there is no `kernels/` directory. The core path is
  SDPA + `torch.compile`, and Inductor already fuses those epilogues.
- A hand-written Turing FlashAttention kernel is out of scope (multi-day effort).

## AI tooling

Design and implementation were driven with Claude (Claude Code). See
[`docs/AI_TOOLS.md`](docs/AI_TOOLS.md) for the prompt → design → diff → verify log.
