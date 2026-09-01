# Ablation study

The delivered speedups isolate the contribution of each optimization. On the
free Kaggle **P100** (sm_60), the graded path is **SDPA in fp32** — `torch.compile`
is unavailable (Triton needs sm≥7.0) and internal fp16 is disabled by default
(it breaks the strict `atol=0.002` gate over 4 layers). So the P100 numbers below
are the **SDPA-alone** contribution vs the explicit-attention baseline.

Toggle stages with environment variables (no code edits):

| Stage | Command (shape 13 example) | Where it helps |
|---|---|---|
| baseline | (reference) `torch_transformer_benchmark.py` | — |
| **+ SDPA** (delivered on P100) | default | fused softmax + fewer passes; big on long S |
| + torch.compile | (auto on sm≥7.0, e.g. T4) | LayerNorm/GELU/bias fusion, CUDA graphs |
| + fp16 autocast | `T3_AUTOCAST=fp16` | tensor cores — **opt-in; can break tol** |
| + shape-14 chunk | `scripts/shape14_optimized_only.py` | fits 16 GB at S=1e5 |

## Measured (Kaggle P100, fp32, SDPA-only)

Median **2.07×** across 13 shapes (min 1.10×, max 4.00×), all PASS, max_abs ≈ 1e-6.

| shape | baseline ms | opt ms | speedup | note |
|---|---|---|---|---|
| 13 (S=1024, long seq) | 168.60 | 42.15 | **4.00×** | SDPA avoids the S² score matrix |
| 11 (16 heads) | 12.55 | 4.91 | 2.56× | fewer per-head passes |
| 12 (S=32, launch-bound) | 3.06 | 1.32 | 2.33× | fused attention → fewer launches |
| 6 (B=10000, throughput) | 772.3 | 418.0 | 1.85× | 1.28M tokens/call |
| 8 (D=1024, GEMM-bound) | 70.93 | 64.59 | 1.10× | dominated by projections; least SDPA benefit |

Full per-shape data: `results.csv`; raw log: `kaggle_p100_run.log`.

## The chunking stage, measured (shape 14)

This is the only stage with a binary outcome rather than a ratio, so it is worth
separating from the speedup table.

| variant | outcome |
|---|---|
| baseline (explicit `[B,H,S,S]` scores) | cannot run — needs ~20.5 TB |
| SDPA + chunks collected into a list + `torch.cat` | **OOM** at the concat: `Tried to allocate 6.10 GiB` |
| SDPA + chunks written into a preallocated output | **runs**: 293377 ms, 10,907 tok/s, peak 14.61 GB, `chunk_bs=1` |

The concat was the whole difference. Collecting 32 chunks and joining them holds
the pieces and the `[32,100000,1024]` result at the same time — a second ~6.5 GB
allocation on a card that already had the 6.5 GB input resident.

## Not measured

`+ torch.compile` has **no numbers here**: the free GPU the Kaggle API allocated
was a P100 (`sm_60`) for every run, and Triton needs `sm>=7.0`, so the compiled
path never activated. The same applies to the true FlashAttention kernel (sm_75+);
SDPA used its memory-efficient backend throughout. Both would only add to the
speedups above, but we are not going to quote a number we did not measure.
