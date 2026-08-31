# Track 3 — Implement a GPU Kernel for a Transformer Layer
### TikTok TechJam 2026 · "Build with Joy, Code for Change"

Optimize the runtime of a Transformer forward pass on a GPU while keeping the
output numerically identical to the reference implementation (per-element
`abs_err ≤ 0.002` **OR** `rel_err ≤ 0.02`), across 14 official test shapes.

## TL;DR — what we do

| Lever | Effect |
|---|---|
| **`F.scaled_dot_product_attention` (FlashAttention)** | Replaces the baseline's `O(S²)` materialized score matrix with an `O(S)` fused kernel. Makes the `seq_len=100000` shape possible at all — the baseline would need **~20.5 TB** just for its attention scores. |
| **Internal fp16 autocast (even under fp32 grading)** | Lights up the GPU tensor cores. `rtol=0.02` (2%) leaves ~40× headroom over fp16 rounding; LayerNorm/softmax reductions stay in fp32. |
| **Self-applied `torch.compile`** | Fuses LayerNorm / bias / GELU epilogues into Triton kernels; independent of the grader passing `--compile-user`. Per-shape mode (CUDA graphs for launch-bound shapes). |
| **Batch chunking (only `seq_len=1e5`)** | Keeps activations inside a 16 GB GPU. |

We keep **every baseline submodule and parameter name unchanged** and rewrite
only the forward compute, so the harness' `copy_model_weights(strict=True)`
succeeds and the comparison is apples-to-apples.

## Files

```
torch_transformer_benchmark.py   official reference & harness (UNMODIFIED)
user_optimized.py                UserOptimizedTransformer — the deliverable
submission.py                    runs the official main() with our model swapped in
run_all.py                       sweeps shapes 1–13 -> results/results.csv
scripts/shape14_optimized_only.py  seq_len=1e5 timing + truncated-S correctness
notebooks/colab_run.ipynb        shapes 1–13 on a free Colab T4
notebooks/kaggle_shape14.ipynb   shape 14 on Kaggle T4/P100
results/  report/  docs/AI_TOOLS.md
```

## Setup

GPU work runs on **free cloud GPUs** (Google Colab T4 / Kaggle T4×2 or P100);
the reference `torch`/`triton` are preinstalled there — do not reinstall.
No local GPU is required (this project was developed on a machine with only an
Intel iGPU; the tech report states the exact cloud environment used).

```bash
# On a Colab/Kaggle GPU runtime, from the repo root:
python submission.py --causal --device cuda --dtype float32 \
  --batch-size 64 --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128
```

## Reproduce the results

```bash
python run_all.py --shapes 1-13 --device cuda --dtype float32   # -> results/results.csv
python scripts/shape14_optimized_only.py --trunc-seq 2048       # shape 14 (Kaggle)
```

`run_all.py` runs each shape in a fresh subprocess and parses the harness'
`summary: PASS/FAIL`, `max_abs`, `max_rel`, and median speedup.

## Results

Measured on a **free Kaggle Tesla P100** (fp32 grading), full data in
[`results/results.csv`](results/results.csv):

**13/13 gradeable shapes PASS · median speedup 1.96× · up to 4.01× · max_abs ≈ 1e-6.**

| # | B,D,H,S,F | pass | max_abs | baseline ms | opt ms | speedup |
|---|---|---|---|---|---|---|
| 1 | 64,128,4,128,128 | ✅ | 1.2e-6 | 5.81 | 3.31 | 1.76× |
| 2 | 1,128,4,128,128 | ✅ | 9.5e-7 | 2.89 | 1.28 | 2.27× |
| 3 | 4,128,4,128,128 | ✅ | 9.5e-7 | 2.95 | 1.50 | 1.96× |
| 4 | 16,128,4,128,128 | ✅ | 1.1e-6 | 2.80 | 1.27 | 2.21× |
| 5 | 128,128,4,128,128 | ✅ | 1.4e-6 | 10.93 | 6.13 | 1.78× |
| 6 | 10000,128,4,128,128 | ✅ | 1.9e-6 | 772.0 | 417.5 | 1.85× |
| 7 | 64,32,4,128,32 | ✅ | 9.5e-7 | 3.94 | 1.93 | 2.05× |
| 8 | 64,1024,4,128,1024 | ✅ | 1.9e-6 | 72.2 | 66.1 | 1.09× |
| 9 | 64,128,1,128,128 | ✅ | 1.2e-6 | 3.76 | 3.05 | 1.23× |
| 10 | 64,128,2,128,128 | ✅ | 1.2e-6 | 4.70 | 3.08 | 1.53× |
| 11 | 64,128,16,128,128 | ✅ | 1.4e-6 | 12.46 | 4.86 | 2.57× |
| 12 | 64,128,4,32,128 | ✅ | 1.2e-6 | 2.84 | 1.26 | 2.25× |
| 13 | 64,128,4,1024,128 | ✅ | 1.4e-6 | 168.4 | 42.0 | **4.01×** |
| 14 | 32,1024,16,100000,1024 | ✅ (truncated) | 1.2e-6 | — baseline infeasible (~20.5 TB) — |

Shape 14's baseline cannot run (its `[B,H,S,S]` scores are ~20.5 TB). Correctness
is validated by construction at a truncated `seq_len`; the full-length forward
needs a GPU with FlashAttention (sm_75+, e.g. T4/A100). On P100 `torch.compile`
is unavailable (Triton needs sm≥7.0), so these numbers are from SDPA alone — a
T4 with compilation enabled is higher still. See `report/figures/`.

## Ablation

`baseline → +SDPA → +fp16 → +torch.compile → +chunk`, toggled via env vars
(no code edits) — see `results/ablation.md`:

```bash
T3_COMPILE=0 T3_AUTOCAST=off python run_all.py --shapes 1   # SDPA only
T3_COMPILE=0                  python run_all.py --shapes 1   # + fp16
                              python run_all.py --shapes 1   # + compile
```

Environment toggles: `T3_AUTOCAST` (auto|fp16|bf16|off), `T3_COMPILE` (1|0),
`T3_COMPILE_MODE`, `T3_FP32_FFN` (1|0), `T3_CHUNK_BS`.

## Correctness notes

- Every element must pass (zero failures); `NaN/Inf` is a hard fail. We target
  `max_abs ≤ 0.001` and `max_rel ≤ 0.01` (half the tolerance) as a safety margin.
- Softmax accumulates in fp32 (SDPA on CUDA), matching the reference; GELU uses
  the exact `approximate="none"` form.
- Shape 14 has no runnable baseline (20.5 TB scores). We validate correctness at
  a truncated `seq_len` where the baseline runs, and argue by construction that
  SDPA is shape-invariant.

## Limitations & future work

- fp16 tolerance is checked empirically per shape; a precision ladder
  (`T3_FP32_FFN`, fp32-SDPA) is available for any shape that fails.
- Hand-written Triton kernels (fused LayerNorm+residual, fused bias+GELU) are an
  optional stretch in `kernels/`; the core path relies on SDPA + `torch.compile`.
- A hand-written Turing FlashAttention kernel is out of scope (multi-day effort).

## AI tooling

Design and implementation were driven with Claude (Claude Code). See
[`docs/AI_TOOLS.md`](docs/AI_TOOLS.md) for the prompt → design → diff → verify log.
