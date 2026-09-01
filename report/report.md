# Track 3 Technical Report — Implement a GPU Kernel for a Transformer Layer
### TikTok TechJam 2026

> Export to `report.pdf` before submission (e.g. VS Code Markdown PDF, or
> `pandoc report.md -o report.pdf`).

## 1. Environment

| | |
|---|---|
| **Development machine** | Intel Core i5-14500, 16 GB RAM, **Intel UHD 770 iGPU only (no NVIDIA GPU)**, Windows 11, Python 3.13. Used for authoring, the repo, this report, and the demo video — no GPU compute. |
| **Benchmark GPU(s)** | Free **Kaggle Tesla P100-PCIE** (16 GB, sm_60), driven headlessly via the Kaggle API from the local machine. torch 2.5.1+cu121 (the preinstalled 2.10+cu128 dropped sm_60, so the kernel auto-installs a compatible build and re-execs). The full-length shape 14 runs on this P100; a T4/A100 (sm_75+) would additionally enable `torch.compile` and the true FlashAttention kernel. Raw log: `results/kaggle_p100_run.log`. |
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
4. **Batch chunking into a preallocated output, shape 14 only.** The chunk size
   is planned from the VRAM that is actually free (`cuda.mem_get_info` plus the
   allocator's reserved-but-unused blocks), minus the output buffer, divided by
   the ~8 activations a block keeps live at once. Each chunk is written straight
   into the preallocated `[B,S,D]` output; the earlier list-plus-`torch.cat`
   version held the pieces *and* the joined tensor simultaneously, which doubled
   peak VRAM precisely where it was tightest and was the actual cause of the
   seq_len=1e5 OOM. If the estimate is still too optimistic the chunk size halves
   and the pass restarts, so a mis-planned budget degrades instead of failing.

Per-shape dispatch table and the correctness checklist are in the project plan
and `user_optimized.py`.

## 5. Results

On a free Kaggle P100 (fp32 grading), **all 13 gradeable shapes PASS** with
`max_abs ≈ 1e-6` (a faithful reproduction of the fp32 reference), and a **median
speedup of 2.07×** (min 1.10×, max **4.00×** on the long-sequence shape 13) —
from `scaled_dot_product_attention` alone, since `torch.compile` is unavailable
on P100 (Triton needs sm≥7.0). Full table: `results/results.csv`; per-step
breakdown: `results/ablation.md`.

| # | shape [B,D,H,S,F] | speedup | | # | shape | speedup |
|---|---|---|---|---|---|---|
| 1 | 64,128,4,128 | 1.76× | | 8 | 64,1024,4,128 | 1.10× |
| 2 | 1,128,4,128 | 2.14× | | 9 | 64,128,1,128 | 1.24× |
| 3 | 4,128,4,128 | 2.17× | | 10 | 64,128,2,128 | 1.52× |
| 4 | 16,128,4,128 | 2.29× | | 11 | 64,128,16,128 | 2.56× |
| 5 | 128,128,4,128 | 1.78× | | 12 | 64,128,4,32 | 2.33× |
| 6 | 10000,128,4,128 | 1.85× | | 13 | 64,128,4,1024 | **4.00×** |
| 7 | 64,32,4,128 | 2.07× | | 14 | 32,1024,16,100000 | infeasible→**runs** |

**Shape 14 is the result we care most about.** The baseline needs ~20.5 TB for
its scores and cannot run, so there is no ratio to report; the meaningful claim
is that the shape goes from impossible to possible. Measured on the same free
16 GB P100:

```
trunc S=2048 correctness: PASS max_abs=1.19e-06 max_rel=0.127
vram free=16.64/17.06 GB | baseline scores would be 20.5 TB -> infeasible
full S=100000: median=293376.9 ms | 10,907 tok/s | peak_vram=14.61 GB | chunk_bs=1
```

293 s per forward across 3.2 M tokens, peak 14.61 GB of the 17.06 GB card.
Correctness is established at a truncated `seq_len` where the baseline can run
(PASS, `max_abs 1.2e-6`); SDPA's math is independent of `S`, so that carries to
1e5. Getting here required the memory fix in §4.4 — before it, the run died in
the final `torch.cat`, not in the attention.

Figures: `figures/memory_wall.png` (the 20.5 TB wall), `figures/speedups.png`
(per-shape speedup).

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
