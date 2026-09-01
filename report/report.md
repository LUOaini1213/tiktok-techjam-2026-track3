# Track 3 Technical Report — Implement a GPU Kernel for a Transformer Layer
### TikTok TechJam 2026

A rendered, shareable version of this report is linked from the repo README.
To produce a PDF locally: open this file in VS Code with the Markdown PDF
extension, or run `pandoc report.md -o report.pdf`.

## 1. Environment

| | |
|---|---|
| **Development machine** | Intel Core i5-14500, 16 GB RAM, **Intel UHD 770 iGPU only (no NVIDIA GPU)**, Windows 11, Python 3.13. Used for authoring, the repo, this report, and the demo video — no GPU compute. |
| **Benchmark GPUs** | Two free Kaggle cards, both driven headlessly via the Kaggle API from the local machine. **Tesla P100-PCIE** (16 GB, sm_60) with torch 2.5.1+cu121 — the preinstalled 2.10+cu128 dropped sm_60, so the kernel detects the mismatch, installs a compatible build and re-execs. **Tesla T4** (16 GB, sm_75) with the preinstalled torch 2.10.0+cu128, no reinstall needed. The API allocates a P100 unless `machine_shape` / `--accelerator NvidiaTeslaT4` asks otherwise — the accepted names appear only in the SDK docstring for `ApiSaveKernelRequest`, and an unrecognised value is silently normalised back to a P100. Raw logs: `results/kaggle_p100_run.log`, `results/kaggle_t4_run.log`, `results/kaggle_t4_fp16_run.log`, `results/kaggle_t4_ablation.log`. |
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

All 13 gradeable shapes PASS on both cards, with `max_abs ≈ 1e-6` — a faithful
reproduction of the fp32 reference, ~1049× inside the `atol=0.002` gate.

| regime | median | range | worst `max_abs` | margin vs `atol` |
|---|---|---|---|---|
| P100, SDPA only | 2.065× | 1.098-4.001× | 1.91e-6 | 1049× |
| **T4, SDPA + compile (shipped)** | **2.357×** | 1.088-4.505× | 1.91e-6 | 1049× |
| T4, + fp16 (`T3_AUTOCAST=fp16`) | 4.014× | 1.320-11.528× | 2.04e-3 | **0.98×** |

| # | shape [B,D,H,S] | P100 | T4 | | # | shape [B,D,H,S] | P100 | T4 |
|---|---|---|---|---|---|---|---|---|
| 1 | 64,128,4,128 | 1.76× | 2.36× | | 8 | 64,1024,4,128 | 1.10× | 1.09× |
| 2 | 1,128,4,128 | 2.14× | 3.40× | | 9 | 64,128,1,128 | 1.24× | 1.28× |
| 3 | 4,128,4,128 | 2.17× | 2.95× | | 10 | 64,128,2,128 | 1.52× | 1.60× |
| 4 | 16,128,4,128 | 2.29× | 2.49× | | 11 | 64,128,16,128 | 2.56× | 3.16× |
| 5 | 128,128,4,128 | 1.78× | 1.98× | | 12 | 64,128,4,32 | 2.33× | 1.95× |
| 6 | 10000,128,4,128 | 1.85× | 1.91× | | 13 | 64,128,4,1024 | **4.00×** | **4.51×** |
| 7 | 64,32,4,128 | 2.07× | 2.72× | | 14 | 32,1024,16,100000 | infeasible→**runs** | |

Two things in that table are worth stating rather than glossing:

**The T4's baselines are slower than the P100's** (shape 13: 316.8 ms vs
168.6 ms; shape 6: 1431.8 ms vs 772.3 ms). The P100 has higher fp32 throughput
and roughly twice the memory bandwidth. Our ratios improve on the T4 anyway,
because the optimized path gains `torch.compile` and the real FlashAttention
backend there while the baseline stays bandwidth-bound. A speedup is a ratio and
it matters which side moved.

**fp16 is nearly twice as fast and we do not ship it.** `T3_AUTOCAST=fp16`
passes all 13 shapes at a median 4.014×. But its worst absolute error,
`max_abs = 0.0020388` on shape 6, has *already crossed* `atol=0.002`; it survives
only because the gate is `abs<=0.002` **OR** `rel<=0.02` and that element's
reference happened to be large enough (`|ref| >= 0.102`) for the relative branch.
Move the same error onto a near-zero reference and the element fails -- and one
failing element fails the shape and forfeits the speed score entirely. We took
the 2.357× that sits 1049× inside tolerance and left fp16 as a documented,
measured flag. This is the one place where our earlier reasoning was wrong: the
repo previously asserted fp16 "breaks the gate", which measurement disproved --
the conclusion survived, the justification did not.

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

**Precision.** The truncated correctness check runs in **fp32**, matching the
graded shapes. The full-length timing runs in **fp16** by necessity: an fp32
input for this shape is 13.1 GB and its output another 13.1 GB, so 26.2 GB is
committed before any activation — more than any free 16 GB GPU has. fp16 halves
that to 13.1 GB and leaves room for the per-chunk working set. So the 293 s /
10,907 tok/s / 14.61 GB numbers are fp16 numbers, and are labelled as such
wherever they appear; the 13 graded shapes in the table above are all fp32.

Figures: `figures/memory_wall.png` (the 20.5 TB wall), `figures/speedups.png`
(per-shape speedup).

## 6. Limitations & what we'd improve with more time

- fp16 is measured, not assumed: 13/13 PASS at 4.014x median, with `max_abs`
  2.04e-3 against a 2e-3 gate. Shipped off; see the ablation for the reasoning.
- `--dtype bfloat16` fails the accuracy gate. This is a property of the
  configuration rather than of our kernel: the reference compared against itself
  recomputed in fp32 fails identically (7603 vs our 6131 elements, same
  `max_abs 0.047`), because bf16's ulp near 1.0 is 0.0078 against `atol=0.002`.
  The graded configuration is fp32.
- The padded fallback still builds a dense `[B,1,S,S]` bias, reintroducing the
  `O(S^2)` allocation SDPA avoids. The graded path (`padding_ratio=0`) never
  takes it; making it memory-efficient is unfinished work.
- Optional hand-written Triton kernels (fused LayerNorm+residual, fused
  bias+GELU) beyond what Inductor already fuses; and a Turing-specific
  FlashAttention kernel (multi-day).

## 7. Reproducibility

`python run_all.py --shapes 1-13` → `results/results.csv`;
`python scripts/shape14_optimized_only.py` for shape 14. Fresh Colab/Kaggle
session, `git clone`, run — numbers reproduce within free-tier variance.

## 8. AI tooling

See `docs/AI_TOOLS.md`.
