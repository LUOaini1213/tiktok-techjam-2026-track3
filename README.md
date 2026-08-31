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

See [`results/results.csv`](results/results.csv) and
[`results/ablation.md`](results/ablation.md). _(Filled in after the cloud runs.)_

| shape | B,D,H,S,L,F | pass | max_rel | speedup |
|---|---|---|---|---|
| … | … | … | … | … |

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
