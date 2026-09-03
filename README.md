# Track 3 — Implement a GPU Kernel for a Transformer Layer
### TikTok TechJam 2026 · "Build with Joy, Code for Change"

Optimize the runtime of a Transformer forward pass on a GPU while keeping the
output numerically identical to the reference implementation (per-element
`abs_err ≤ 0.002` **OR** `rel_err ≤ 0.02`), across 14 official test shapes.

## TL;DR — what we do

| Lever | Effect |
|---|---|
| **`F.scaled_dot_product_attention` (memory-efficient fused attention)** | Replaces the baseline's `O(S²)` materialized score matrix with an `O(S)` fused kernel. Makes the `seq_len=100000` shape possible at all — the baseline would need **~20.5 TB** just for its attention scores. Measured: the full 100k-token forward completes in **14.6 GB** on a free P100. |
| **Internal fp16 autocast** (`T3_AUTOCAST=fp16`, **opt-in**) | Lights up the tensor cores and roughly **doubles** the median speedup (2.28x -> 4.01x). Shipped **off**: it passes all 13 shapes, but its worst absolute error has already crossed `atol=0.002` and survives on the relative branch alone. Reductions stay in fp32. |
| **Self-applied `torch.compile`** | Fuses LayerNorm / bias / GELU epilogues into Triton kernels; independent of the grader passing `--compile-user`. Needs sm≥7.0, so it is inactive on a P100. Measured on a T4 it is worth +0.3x on compute-bound shapes and a **net loss** on launch-bound ones — so the model now times eager against compiled once, on the real input, and keeps the winner (`T3_COMPILE=auto`, the default). |
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
| **Tesla T4 (`sm_75`)** | **SDPA + autotuned `torch.compile`** | **2.280x** | 1.090x - 4.541x | 1.91e-6 |
| Tesla T4, `T3_AUTOCAST=fp16` | + fp16 tensor cores | 4.014x | 1.320x - 11.528x | 2.04e-3 — *see below* |

Kaggle's API hands out a P100 unless you ask otherwise, which is why the first
row exists: on `sm_60` Triton will not build, so `torch.compile` never engages.
Pass `--accelerator NvidiaTeslaT4` and compilation lights up. The attention
kernel itself is the **same on both cards** — see the next section.

Per shape, fp32 (`results/results.csv` = P100, `results/results_t4.csv` = T4):

| # | B,D,H,S,F | P100 base -> opt | P100 | T4 base -> opt | T4 |
|---|---|---|---|---|---|
| 1 | 64,128,4,128,128 | 5.91 -> 3.37 | 1.75x | 9.45 -> 4.14 | 2.28x |
| 2 | 1,128,4,128,128 | 3.16 -> 1.48 | 2.14x | 3.37 -> 0.92 | 3.65x |
| 3 | 4,128,4,128,128 | 3.18 -> 1.46 | 2.17x | 3.45 -> 1.10 | 3.14x |
| 4 | 16,128,4,128,128 | 3.15 -> 1.38 | 2.29x | 3.57 -> 1.23 | 2.91x |
| 5 | 128,128,4,128,128 | 11.00 -> 6.19 | 1.78x | 18.30 -> 9.20 | 1.99x |
| 6 | 10000,128,4,128,128 | 772.29 -> 417.97 | 1.85x | 1436.28 -> 749.81 | 1.92x |
| 7 | 64,32,4,128,32 | 4.05 -> 1.96 | 2.06x | 6.36 -> 2.35 | 2.71x |
| 8 | 64,1024,4,128,1024 | 70.93 -> 64.59 | 1.10x | 132.13 -> 121.25 | 1.09x |
| 9 | 64,128,1,128,128 | 3.85 -> 3.11 | 1.24x | 6.41 -> 5.11 | 1.25x |
| 10 | 64,128,2,128,128 | 4.77 -> 3.13 | 1.52x | 7.98 -> 5.13 | 1.56x |
| 11 | 64,128,16,128,128 | 12.54 -> 4.91 | 2.56x | 22.47 -> 7.22 | 3.11x |
| 12 | 64,128,4,32,128 | 3.06 -> 1.32 | 2.33x | 3.43 -> 1.51 | 2.27x |
| 13 | 64,128,4,1024,128 | 168.60 -> 42.15 | 4.00x | 317.90 -> 70.00 | 4.54x |

One honest observation from that table: **the T4's baselines are slower than the
P100's** (shape 13: 317.9 ms vs 168.6 ms). The P100 has more fp32 throughput and
about twice the memory bandwidth. The T4 ratios are better anyway because our
path picks up compile there while the baseline stays
bandwidth-bound. A speedup is a ratio; it is worth saying which side moved.

### Which attention kernel actually ran — measured, not assumed

Earlier drafts of this README said "FlashAttention". We probed it: force each
SDPA backend alone, for every dtype and head_dim the sweep uses, on both cards
(`results/kaggle_t4_probe.log`, `results/kaggle_p100_probe.log`). Identical on
the T4 and the P100:

| dtype | head_dim | flash | efficient | math |
|---|---|---|---|---|
| fp32 | 8 / 32 / 64 / 128 / 256 | **no** | yes | yes |
| fp16 | 8 / 32 / 64 / 128 / 256 | **no** | yes | yes |

PyTorch's flash backend is fp16/bf16-only and, in current releases, needs sm_80+;
the graded path is fp32 on sm_60 / sm_75 cards. **No run in this project used
FlashAttention.** Every `scaled_dot_product_attention` call went through the
memory-efficient backend — which is the kernel with `O(S)` memory and a fused
softmax, i.e. the property the shape-14 result actually depends on. The name was
wrong; the mechanism was not. It also means the T4-over-P100 gain is compilation
plus hardware, not a different attention kernel.

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

**And the obvious middle ground does not exist.** The natural next question is a
mixed assignment — fp16 where the tensor cores pay, fp32 where the error accumulates.
`T3_AUTOCAST=fp16 T3_FP32_FFN=1` is exactly that (fp16 attention projections and
SDPA, fp32 FFN and LayerNorm), and measured it lands at **2.953x median with a
worst `max_abs` of 1.72e-03** — a margin of **1.17x**, every shape between
8.48e-04 and 1.72e-03. Moving the FFN and the norms to fp32 bought 2.04e-3 → 1.72e-03,
which says the error floor lives in the fp16 attention matmuls themselves, not in
what follows them. Three regimes measured; only fp32 has a margin worth the word
(`results/results_t4_mixed.csv`, `results/kaggle_t4_mixed_run.log`).

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

293 s per forward over 3.2 M tokens, peak 14.6 GB. On a **T4** — same memory-efficient SDPA backend, but its fp16 tensor cores now
carry the natively-fp16 matmuls — the same forward takes **204 s at 15,676 tok/s**
with a 14.58 GB peak — on a card that has only 15.64 GB in total, tighter than the
P100's 17.06 GB, and it still fits (`results/kaggle_t4_shape14.log`). Correctness is validated at a
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

One caveat we would rather state than hide: the P100 is `sm_60`, so it gets no
`torch.compile` (Triton needs sm>=7.0); every P100 speedup is **from SDPA alone**.
The attention kernel is the same memory-efficient one on both cards — PyTorch's
flash backend is fp16-only and needs sm_80+, so neither GPU could run it (probe
below). See `report/figures/`.

## Ablation

Every stage is selected by environment variable with **no code edits**, so the
ablation and the delivered path are literally the same code. The measured
4-shape x 4-stage table is in [`results/ablation.md`](results/ablation.md)
(machine-readable: `results/ablation_t4.csv`).

Shipped defaults are `T3_AUTOCAST=off`, `T3_COMPILE=auto`: SDPA, plus
compilation wherever a first-forward timing on the real input says it wins.
On the T4 that was 7 of the 11 shapes small enough to tune; eager won shapes
5, 8, 11 and 12, and shape 12 gained 15% for it (1.971x -> 2.271x).

**Fused QKV projection (`T3_FUSED_QKV=1`), measured and declined.** One `[3D, D]`
GEMM instead of three `[D, D]` ones: two fewer launches and the activation read
once. On the T4: median 2.387x against 2.280x, but that is one shape (the
median element) moving inside run-to-run noise; the mean is flat (2.490x vs
2.494x), 8 shapes better and 5 worse by amounts the small shapes
swing between identical runs, and shape 6 — where reading the activation once
should pay most — did not move. The likely reason is that the efficient
attention backend re-copies the strided q/k/v views the split produces, handing
back the reads the fusion saved. Flag kept, off (`results/results_t4_qkv.csv`). Pass `--out` so a stage run does not
overwrite the committed `results/results.csv`:

```bash
T3_COMPILE=0 T3_AUTOCAST=off  python run_all.py --shapes 1 --out /tmp/sdpa.csv
T3_COMPILE=1 T3_AUTOCAST=off  python run_all.py --shapes 1 --out /tmp/compile.csv
T3_COMPILE=0 T3_AUTOCAST=fp16 python run_all.py --shapes 1 --out /tmp/fp16.csv
T3_COMPILE=1 T3_AUTOCAST=fp16 python run_all.py --shapes 1 --out /tmp/both.csv
```

Environment toggles: `T3_AUTOCAST` (auto|fp16|bf16|off), `T3_COMPILE` (1|0),
`T3_COMPILE_MODE`, `T3_FP32_FFN` (1|0), `T3_CHUNK_BS`, `T3_TRITON` (1|0).

## The hand-written Triton kernel

`kernels/fused_layernorm.py` is a fused **residual-add + LayerNorm** written in
Triton. The fusion target was chosen deliberately: `nn.LayerNorm` is already a
tuned CUDA kernel and reimplementing it alone is a predictable loss, but eager
PyTorch does **not** fuse the pre-norm pattern a Transformer block repeats twice
per layer —

```python
x = x + sublayer(norm(x))     # the add is one kernel, the norm is another
```

— and each of those touches the whole `[B,S,D]` activation. Fusing them turns
four passes over that tensor into two. Reductions accumulate in fp32 regardless
of storage dtype, so the fusion spends none of the accuracy budget.

**At the operator level it wins.** Measured on a T4 at the real activation sizes
(`results/triton_bench_t4.csv`, raw log `results/kaggle_t4_triton_bench.log`):

| case | rows × D | eager ms | Inductor ms | **Triton ms** | vs eager | vs Inductor |
|---|---|---|---|---|---|---|
| shape 6 | 1.28M × 128 | 20.660 | 11.004 | **10.763** | **1.92×** | 1.02× |
| shape 13 | 65536 × 128 | 1.082 | 0.657 | **0.602** | **1.80×** | 1.09× |
| shape 1/5/9–11 | 8192 × 128 | 0.236 | **0.148** | 0.159 | 1.49× | 0.93× |
| shape 7 | 8192 × 32 | 0.092 | 0.102 | **0.076** | 1.21× | **1.34×** |
| shape 8 | 8192 × 1024 | 0.719 | 0.673 | **0.649** | 1.11× | 1.04× |
| shape 2 | 128 × 128 | **0.050** | 0.114 | 0.086 | 0.58× | 1.32× |

It beats eager on 5 of 6 and Inductor on 5 of 6, with `max_abs ≤ 1.43e-6`. The
one real loss to eager is the 128-row case, and it is explainable rather than
mysterious: 128 rows is 128 Triton programs, which does not fill a T4, while
eager's two kernels are each small enough that launching two of them is cheaper
than under-occupying one.

**End to end, as a raw launch, it was a net loss: 1.929× against 2.282×.**
A raw Triton call breaks a `torch.compile` graph, so the first version had to
switch compilation off to run at all, trading one won fusion for every fusion
Inductor was doing elsewhere.

**So we did the fix.** The kernel is now registered through
`torch.library.triton_op` (`exactswap::fused_add_layernorm`), with
`wrap_triton` letting the compiler trace the launch under FakeTensor mode, so
Inductor schedules it *inside* the compiled graph. `T3_TRITON=1` no longer turns
compilation off. A second run, all columns from the same session
(`results/triton_bench_t4.csv`, `results/kaggle_t4_triton_bench.log`):

| case | rows × D | eager | Inductor | registered op, eager | **op inside Inductor's graph** | in-graph vs Inductor |
|---|---|---|---|---|---|---|
| shape 6 | 1.28M × 128 | 20.043 | 11.080 | 10.592 | **10.594** | **1.046×** |
| shape 13 | 65536 × 128 | 1.055 | 0.657 | 0.634 | **0.633** | **1.038×** |
| shape 8 | 8192 × 1024 | 0.734 | 0.674 | 0.666 | **0.666** | 1.012× |
| shape 2 | 128 × 128 | 0.049 | 0.108 | 0.129 | 0.107 | 1.008× |
| shape 7 | 8192 × 32 | 0.095 | **0.108** | 0.125 | 0.125 | 0.862× |
| shape 1/5/9–11 | 8192 × 128 | 0.236 | **0.150** | 0.186 | 0.183 | 0.819× |

**End to end: 1.929× → 2.190× with registration, against 2.282× with compilation fixed on** (the autotune default now ships at 2.280×, within noise of that),
still 13/13 PASS (`results/results_t4_triton.csv`). The mechanism was right —
composing with the compiler recovered most of the loss — and it still does not
win, for two reasons the table makes visible:

- **Inductor's own fusion of these two ops is within ±5% of the hand-written
  kernel** on the memory-bound shapes (1.01–1.05× in our favour), which is to say
  the compiler already writes this kernel about as well as we did.
- **On the small and narrow shapes Inductor wins outright** (0.82–0.86×): it
  fuses *across* op boundaries, and a custom op is an opaque wall it cannot see
  through. Registration also costs 30–70 µs of dispatcher overhead per call —
  compare the registered-op column against the raw-launch table above on the
  128-row and 8192×32 cases — which only launch-bound shapes notice.

We matched the compiler. We did not beat it. The kernel stays behind
`T3_TRITON`, correct and composable, with every number published; the shipped
path is the one that is faster.

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
  margin back on a specific shape. The mixed assignment (fp16 attention, fp32 FFN and LayerNorm) is measured too:
  2.953x at `max_abs` 1.72e-03, margin 1.17x — the error floor is in the
  attention matmuls, so there is no cheap middle ground on this axis. A real one
  would need fp16 storage with fp32 accumulation *inside* the attention kernel,
  which SDPA's fp16 path does not expose.
- The stage ablation covers 4 representative shapes on one GPU, one run each;
  the 13-shape sweeps are the ones with repeat trials behind them.
- The fused bias+GELU epilogue kernel was scoped and not built; Inductor
  already fuses it. The fused add+LayerNorm kernel **was** built and measured —
  see below — and is off by default for a reason we can name.
- A hand-written attention kernel for Turing (fp16 storage, fp32 accumulation) is
  out of scope (multi-day effort); it is also the only thing that could open the
  precision middle ground measured above.
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

## Demo video

**Watch it: https://youtu.be/3aAw-jq1oTM**

`build/track3_demo.mp4` (**exactly 3:00**, 1920x1080, subtitles in `.srt`) is
**generated** from the measured CSVs, not hand-edited:

```bash
python scripts/make_video.py --out build/track3_demo.mp4
```

It follows the required submission structure, and the section windows are fixed
rather than fitted to whatever the narration happens to run to:

| | | |
|---|---|---|
| 0:00–0:15 | Problem | the task, the per-element gate, the 20.5 TB shape |
| 0:15–0:35 | Our Solution | fused attention, self-applied compile, VRAM-planned chunking |
| 0:35–0:55 | Architecture | baseline vs our data flow, and where the S×S tensor dies |
| **0:55–2:20** | **Live Demo** | build → push → T4 comes up → 13 shapes → shape 14 → summary |
| 2:20–2:45 | Results | speedup chart, the three precision regimes |
| 2:45–3:00 | Impact | drop-in reuse, free-tier reproducibility |

If a section's narration overruns its window the script re-synthesizes it at a
higher speaking rate rather than letting the timeline drift, and reports the rate
it used. The demo block is a **replay of the recorded run**, labelled as such on
screen and sourced from `results/kaggle_t4_run.log` and
`results/kaggle_t4_shape14.log` — the GPUs are in Kaggle, so there is no local
screen to capture. `--no-vo` renders a silent, subtitle-only cut with no network
access.

## Report

A rendered version of the technical report, with the figures inline:
**https://claude.ai/code/artifact/80227d3c-9682-42d0-a957-bf5188704088**
(print to PDF from the browser). Source: [`report/report.md`](report/report.md);
the page is assembled from the measured CSVs by
`scripts/build_report_page.py`, so its numbers cannot drift from the data.

## AI tooling

Design and implementation were driven with Claude (Claude Code). See
[`docs/AI_TOOLS.md`](docs/AI_TOOLS.md) for the prompt → design → diff → verify log.
