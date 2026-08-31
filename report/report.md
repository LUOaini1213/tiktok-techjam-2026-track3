# Track 3 Technical Report — Implement a GPU Kernel for a Transformer Layer
### TikTok TechJam 2026

> Export to `report.pdf` before submission (e.g. VS Code Markdown PDF, or
> `pandoc report.md -o report.pdf`).

## 1. Environment

| | |
|---|---|
| **Development machine** | Intel Core i5-14500, 16 GB RAM, **Intel UHD 770 iGPU only (no NVIDIA GPU)**, Windows 11, Python 3.13. Used for authoring, the repo, this report, and the demo video — no GPU compute. |
| **Benchmark GPU(s)** | Google Colab free **NVIDIA T4** (16 GB, Turing sm_75) for shapes 1–13; Kaggle **T4 / P100** (16 GB) for shape 14. torch `<fill>`, CUDA `<fill>`. |
| **Why cloud** | Track 3 is a GPU-kernel task; `CUDA`/`Triton`/tensor cores require an NVIDIA GPU. Free cloud GPUs (Colab is an explicitly allowed dev tool) were used and are reported honestly here. |

## 2. The problem and the grading contract

The harness compares a `UserOptimizedTransformer` against the reference
`BaselineTransformer` across 14 shapes. An entry passes a trial only if **every**
output element satisfies `abs_err ≤ 0.002` **OR** `rel_err ≤ 0.02`, and any
`NaN/Inf` is a hard fail. Only if accuracy passes is the median latency timed
(`speedup = baseline.median / optimized.median`). Weights are copied with
`strict=True`, so parameter names must match exactly.

## 3. Key insight: the memory wall at S=100000

The baseline computes attention explicitly: `scores = Q·Kᵀ` of shape
`[B, H, S, S]`. For shape 14 that is `32 · 16 · 10⁵ · 10⁵ = 5.12×10¹²` elements
≈ **20.5 TB** in fp32 — impossible on any GPU. The baseline therefore cannot run
shape 14 at all; only a memory-efficient (FlashAttention-style) attention with
`O(S)` memory can. This is the crux of the task.

## 4. Optimizations

1. **SDPA / FlashAttention.** `F.scaled_dot_product_attention(is_causal=True,
   attn_mask=None)` on the no-padding hot path. `O(S)` memory, fused softmax,
   tensor-core matmuls. Unlocks shape 14 and wins big on long-sequence shape 13.
2. **Internal fp16 autocast under fp32 grading.** Tensor cores on the T4 (which
   has fp16 MMA but no bf16/TF32). Reductions kept in fp32; `rtol=0.02` leaves
   ~40× margin over fp16 rounding. Verified per shape.
3. **Self-applied `torch.compile`.** Inductor fuses LayerNorm/bias/GELU
   epilogues into Triton kernels; `reduce-overhead` (CUDA graphs) for
   launch-bound small shapes, `default` otherwise. Independent of `--compile-user`.
4. **Batch chunking for shape 14 only.** Keeps activations within 16 GB; outputs
   concatenated in order.

Per-shape dispatch table and the correctness checklist are in the project plan
and `user_optimized.py`.

## 5. Results

Insert `results/results.csv` as a table and `results/ablation.md`. Headline:
**median speedup `<fill>`× across shapes 1–13, all PASS**, and shape 14
**runs in `<fill>` ms / `<fill>` tok/s where the baseline is infeasible.**

_(figures: memory-wall bar chart `figures/memory_wall.png`, per-shape speedup
`figures/speedups.png`.)_

## 6. Limitations & what we'd improve with more time

- Empirical fp16 tolerance per shape; a precision ladder handles any failure.
- Optional hand-written Triton kernels (fused LayerNorm+residual, fused
  bias+GELU) beyond what Inductor already fuses; and a Turing-specific
  FlashAttention kernel (multi-day).

## 7. Reproducibility

`python run_all.py --shapes 1-13` → `results/results.csv`;
`python scripts/shape14_optimized_only.py` for shape 14. Fresh Colab/Kaggle
session, `git clone`, run — numbers reproduce within free-tier variance.

## 8. AI tooling

See `docs/AI_TOOLS.md`.
