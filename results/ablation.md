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

Median **1.96×** across 13 shapes (min 1.09×, max 4.01×), all PASS, max_abs ≈ 1e-6.

| shape | baseline ms | opt ms | speedup | note |
|---|---|---|---|---|
| 13 (S=1024, long seq) | 168.41 | 42.01 | **4.01×** | SDPA avoids the S² score matrix |
| 11 (16 heads) | 12.46 | 4.86 | 2.57× | fewer per-head passes |
| 2 (B=1, launch-bound) | 2.89 | 1.28 | 2.27× | fused attention → fewer launches |
| 6 (B=10000, throughput) | 772.0 | 417.5 | 1.85× | 1.28M tokens/call |
| 8 (D=1024, GEMM-bound) | 72.23 | 66.12 | 1.09× | dominated by projections; least SDPA benefit |

Full per-shape data: `results.csv`.

## Pending (T4, sm_75) — appended after the Colab run
`+ torch.compile` (LayerNorm/GELU/bias fusion, CUDA graphs) is expected to lift
the launch-bound and pointwise-heavy shapes further, and FlashAttention makes
shape 14 run at full length. Numbers to be filled from the T4 sweep.
