# Devpost 提交交接单 — TikTok TechJam 2026

**给代填的人：** 下面每一项都是照抄即可，不需要理解内容。按顺序填完点提交。
遇到没列出的可选字段，留空即可。

提交页面：TikTok TechJam 2026 on Devpost → 我的项目 → Edit

---

## 1. Track（赛道选择）

```
Track 3 — Implement a GPU Kernel for a Transformer Layer
```

---

## 2. Project name（项目名）

```
ExactSwap — Drop-in GPU Transformer Layer
```

---

## 3. Elevator pitch（一句话简介，199 字符）

```
A drop-in Transformer layer that is 2.29x faster across all 13 graded shapes while staying 1049x inside the tolerance gate — and runs the 100,000-token shape whose reference needs 20.5 TB and cannot.
```

---

## 4. Public code repository（公开代码仓库）

```
https://github.com/LUOaini1213/tiktok-techjam-2026-track3
```

---

## 5. Video demo link（演示视频）

```
https://youtu.be/3aAw-jq1oTM
```

---

## 6. "Try it out" links（试用链接）

```
https://github.com/LUOaini1213/tiktok-techjam-2026-track3
```

---

## 7. Built with（技术标签，共 24 个，逗号分隔粘贴）

```
python, pytorch, cuda, triton, memory-efficient-attention, scaled-dot-product-attention, torch-compile, torch-inductor, cuda-graphs, gpu, kernel-optimization, mixed-precision, fp16, memory-optimization, transformer, attention, kaggle, tesla-t4, tesla-p100, matplotlib, pillow, edge-tts, moviepy, claude-code
```

---

## 8. Upload a File（上传文件，限 35 MB）

上传这个文件（4.6 MB）：

```
track3-submission.zip
```

---

## 9. Image gallery（图片画廊，最多 15 张）

上传这五张，**`01_results.png` 必须放第一张**（它是封面缩略图）：

```
01_results.png
02_memory_wall.png
03_architecture.png
04_precision.png
05_demo.png
```

---

## 10. About the project（项目正文）

**Devpost 的正文编辑器支持 Markdown 和 LaTeX 数学公式。把下面分隔线之间的全部内容
原样复制粘贴进去** —— 包括 `$...$` 和 `$$...$$`，Devpost 会把它们渲染成公式。

> 粘贴后请**扫一眼公式有没有渲染出来**。万一没渲染（显示成 `$...$` 原文），
> 内容依然读得通 —— 每个公式旁边的句子都自带解释，不存在「公式挂了就看不懂」的段落。
> 真要救急，把 `$$...$$` 整段删掉也不影响文章成立。

---8<--- 从这里开始复制 ---8<---

## Inspiration

The task reads like a speed contest. It is a correctness contest with a speed prize attached.

The grader checks **every single output element**, and an element passes only if

$$|o_i - r_i| \le 0.002 \quad\textbf{or}\quad |o_i - r_i| \le 0.02\,|r_i|$$

One failing element fails the whole shape, and a failed shape forfeits its speed score entirely. So the interesting question was never "how fast can this go" — it was "how fast can this go while the answer provably does not move".

Then we read the shape list. Shape 14 asks for $S = 100{,}000$, and the reference implementation materializes its attention scores explicitly as a $[B, H, S, S]$ tensor:

$$32 \cdot 16 \cdot (10^5)^2 \cdot 4\ \text{bytes} \;=\; 2.05 \times 10^{13}\ \text{bytes} \;\approx\; \mathbf{20.5\ TB}$$

No GPU holds that. The baseline does not run *slowly* on shape 14 — it does not run. That is the shape we wanted.

## What it does

**ExactSwap** is a drop-in replacement for the reference `BaselineTransformer`. It subclasses it, keeps every submodule and parameter name, and rewrites only the forward compute — so the official `copy_model_weights(..., strict=True)` succeeds and the comparison is apples to apples.

**On a free Kaggle Tesla T4 under fp32 grading: 13/13 gradeable shapes PASS, median speedup 2.286× (range 1.094×–4.436×), worst absolute error $1.91\times10^{-6}$.** That is a factor of

$$\frac{0.002}{1.91 \times 10^{-6}} \approx 1049$$

inside the tolerance gate.

**And shape 14 runs.** The full 100,000-token forward completes in **204 s at 15,676 tokens/s, peaking at 14.58 GB** — on a card that only has 15.64 GB in total. Against a reference needing 20.5 TB, the meaningful result is not a ratio; it is the difference between *cannot run* and *runs*.

Three levers, each selectable by environment variable with no code edits, so the ablation and the delivered path are literally the same code:

1. **`F.scaled_dot_product_attention`** (memory-efficient backend) — $O(S)$ memory instead of the baseline's $O(S^2)$. The score matrix is never built. Causality goes through `is_causal=True`, generated inside the kernel, because a dense $[S,S]$ mask at $S = 10^5$ is $10^{10}$ bytes on its own.
2. **Self-applied `torch.compile`** — the model compiles itself on the first forward, so the speedup does not depend on the grader passing `--compile-user`. Requires $\text{sm} \ge 7.0$.
3. **VRAM-planned batch chunking** — the chunk size is solved for at runtime rather than tuned:

$$n_{\text{chunk}} \;=\; \left\lfloor \frac{0.6\,\bigl(M_{\text{free}} - M_{\text{out}}\bigr)}{b \cdot k \cdot S \cdot \max(D, F)} \right\rfloor$$

with $M_{\text{free}}$ from `cuda.mem_get_info` plus the allocator's reserved-but-unused blocks, $b$ bytes per element, and $k \approx 8$ live intermediates per block. If that estimate is still optimistic, the chunk halves and the pass restarts.

## How we built it

Development happened on a laptop with an Intel iGPU and **no NVIDIA card**. Every GPU run was pushed headlessly to Kaggle and pulled back: a builder inlines the unmodified harness, the optimized model and a sweep driver into a single self-contained kernel, so nothing needs to be attached to the notebook.

**Three results that were not obvious going in:**

**1. The shape-14 OOM was not in the attention.** After switching to fused attention, shape 14 still died with `Tried to allocate 6.10 GiB`. The fused kernel was fine. The failure was the last line: chunking the batch and joining the pieces with `torch.cat` holds the chunks **and** the joined $[32, 10^5, 1024]$ result simultaneously — a second $\approx 6.5$ GB allocation on a card that already had the 6.5 GB input resident. Writing each chunk straight into a preallocated output removed the copy, and the shape ran. The bug was one line away from where it appeared.

That run is fp16, and for this shape that is forced rather than chosen. In fp32 the input and output alone are

$$2 \cdot 32 \cdot 10^5 \cdot 1024 \cdot 4\ \text{bytes} \;=\; 26.2\ \text{GB} \;>\; 15.64\ \text{GB}$$

committed before a single activation. Correctness for shape 14 is therefore established separately, in **fp32**, at a truncated $S$ where the reference still fits — PASS at `max_abs` $= 1.19\times10^{-6}$ — and SDPA's mathematics does not depend on $S$.

**2. The Kaggle API silently gives you the wrong GPU.** `torch.compile` and fp16 tensor cores were both marked "not measured" for most of this project, because the API-allocated card is a Tesla **P100** (`sm_60`) where Triton will not build. The kernels API *does* let you choose — `machine_shape` / `--accelerator` — but the accepted values (`NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`) appear only in the SDK docstring for `ApiSaveKernelRequest`, and **anything unrecognised is silently normalised back to a P100**. Our first two guesses looked like successful requests and were not. We confirmed the right one by pushing a throwaway kernel that printed `get_device_name`.

**3. Two GPUs make an accidental ablation.** The same code scores 2.065× on the P100 (SDPA alone) and 2.286× on the T4 (SDPA + first-forward autotune). Worth stating plainly: the T4's *baselines* are **slower** than the P100's — 318.7 ms against 168.6 ms on shape 13 — because the P100 has more fp32 throughput and roughly twice the memory bandwidth. Our ratio improves on the T4 anyway, because the optimized path gains compilation there while the baseline stays bandwidth-bound. A speedup is a ratio, and it matters which side moved. One more correction we owe: earlier drafts credited the T4 with "FlashAttention". We probed every SDPA backend for every dtype and head dimension on both cards, and **flash was never available** — it is fp16-only and needs sm_80+. Every run used the memory-efficient backend, whose $O(S)$ memory is the property shape 14 depends on. Right mechanism, wrong name, now fixed.

**4. We wrote the kernel, and it lost where it counted.** The track is named "implement a GPU kernel", so composing `scaled_dot_product_attention` with `torch.compile` — however well measured — leaves an obvious gap. `kernels/fused_layernorm.py` closes it: a fused **residual-add + LayerNorm** in Triton. The target was picked on purpose. `nn.LayerNorm` alone is already a tuned CUDA kernel and rewriting it is a predictable loss; what eager PyTorch does *not* fuse is the pre-norm pattern a block repeats twice per layer, $x = x + \mathrm{sublayer}(\mathrm{norm}(x))$, where the add and the norm each traverse the full $[B, S, D]$ activation. Fusing them takes four passes down to two.

**At the operator level it wins.** On a T4, at the real activation sizes:

| case | rows $\times$ D | eager | Inductor | **Triton** | vs eager | vs Inductor |
|---|---|---|---|---|---|---|
| shape 6 | 1.28M $\times$ 128 | 20.660 ms | 11.004 ms | **10.763 ms** | **1.92×** | 1.02× |
| shape 13 | 65536 $\times$ 128 | 1.082 ms | 0.657 ms | **0.602 ms** | **1.80×** | 1.09× |
| shape 7 | 8192 $\times$ 32 | 0.092 ms | 0.102 ms | **0.076 ms** | 1.21× | **1.34×** |
| shape 2 | 128 $\times$ 128 | **0.050 ms** | 0.114 ms | 0.086 ms | 0.58× | 1.32× |

Five of six against eager, five of six against Inductor, at `max_abs` $\le 1.43\times10^{-6}$. The single real loss is the 128-row case, and it is explainable rather than mysterious: 128 rows is 128 Triton programs, which does not fill a T4, while eager's two kernels are each small enough that launching two beats under-occupying one.

**End to end, as a raw launch, it was a net loss.** With `T3_TRITON=1` the sweep scored **1.929×** median against the shipped **2.282×**, because a raw Triton call breaks a `torch.compile` graph, so enabling the kernel turned compilation off and traded *one* fusion we win for *every* fusion Inductor was doing elsewhere.

**So we did the fix.** Registered through `torch.library.triton_op`, with `wrap_triton` letting the compiler trace the launch, the kernel now runs *inside* Inductor's graph and compilation stays on. Same session, all columns:

| case | rows $\times$ D | Inductor | **our op in-graph** | ratio |
|---|---|---|---|---|
| shape 6 | 1.28M $\times$ 128 | 11.080 ms | **10.594 ms** | **1.046×** |
| shape 13 | 65536 $\times$ 128 | 0.657 ms | **0.633 ms** | **1.038×** |
| shape 8 | 8192 $\times$ 1024 | 0.674 ms | **0.666 ms** | 1.012× |
| shape 7 | 8192 $\times$ 32 | **0.108 ms** | 0.125 ms | 0.862× |
| shape 1 | 8192 $\times$ 128 | **0.150 ms** | 0.183 ms | 0.819× |

End to end: $1.929\times \to 2.190\times$ with registration, against $2.282\times$ with compilation fixed on — still 13/13 PASS. The mechanism was right; composing with the compiler recovered most of the loss. And it still does not win, for two reasons the table makes visible: Inductor's own fusion of these two ops is within $\pm 5\%$ of the hand-written kernel on the memory-bound shapes — the compiler already writes this kernel about as well as we did — and on the small, narrow shapes it wins outright, because it fuses *across* op boundaries where a custom op is an opaque wall, and because registration costs 30–70 µs of dispatch per call that only launch-bound shapes notice.

**We matched the compiler. We did not beat it.** The kernel exists, is correct, composes with `torch.compile`, and ships disabled — with all of those numbers published rather than the flattering one.

## Challenges we ran into

**We shipped a claim we had never measured, and measuring it proved us wrong.** The repository asserted for most of its life that internal fp16 "breaks the strict $\texttt{atol}=0.002$ gate". Once we had a T4 and ran it, fp16 **passed all 13 shapes**, at a median **4.014×** — nearly double the shipped path. The assertion was simply false.

The conclusion survived anyway, for a better reason. fp16's worst absolute error is $2.0388 \times 10^{-3}$ on shape 6, which has **already crossed** $\texttt{atol} = 2 \times 10^{-3}$. It passes only via the second branch of the gate, which requires

$$0.02\,|r_i| \;\ge\; 2.0388\times10^{-3} \quad\Longrightarrow\quad |r_i| \;\ge\; 0.102$$

— that element happened to have a large enough reference value. Nothing makes that repeatable: put the same error on a near-zero reference and the element fails, and one failing element forfeits the shape. The trade is therefore

$$\underbrace{2\times \text{ speed}}_{\text{4.014}\times\text{ vs }2.286\times} \quad\text{for}\quad \underbrace{\frac{0.002}{2.0388\times10^{-3}} \approx 0.98}_{\text{margin, i.e. none}} \quad\text{instead of}\quad 1049$$

We took the margin and left fp16 as a documented, measured flag. **The conclusion was right and the reasoning behind it was wrong, and only running it revealed which.**

**We ran an adversarial audit against our own documentation, and it found four wrong claims.** Multiple agents cross-checked every number in the README, report, ablation and Devpost draft against `results/*.csv` and the raw kernel logs, and every finding was independently verified before being accepted. Confirmed and fixed: shape 14's headline numbers were fp16 while every document framed the run as fp32; the Devpost draft credited speedups to "tensor cores + compilation" on a run whose GPU had neither; a "$\approx 2000\times$ inside tolerance" figure was arithmetic on a number no shape actually hit (the real margin is $\approx 1049\times$); and the README pointed at a `kernels/` directory that never existed. The audit also found a real bug in our own tooling — the log parser raised `NameError` on any log containing a shape-level error — and, with some irony, one audit agent triggered the exact hazard it had just reported, by running the sweep on this CPU-only machine and overwriting the committed GPU results. Nothing reached the remote, the file regenerates from the committed log exactly, and the script now refuses that write.

**Two findings turned out not to be defects, and proving that mattered.** `--dtype bfloat16` fails the gate: 6131 of 131072 elements, `max_abs` $= 0.047$. So does **the reference compared against itself** recomputed in fp32 — 7603 elements, identical `max_abs`. The reason is arithmetic, not implementation:

$$\mathrm{ulp}_{\text{bf16}}(1.0) = 2^{-7} \approx 0.0078 \;\;\gg\;\; \texttt{atol} = 0.002$$

bf16 cannot represent the gate's own precision, so no implementation that reorders a single operation can hold it. That is a property of the configuration; the graded configuration is fp32.

## Accomplishments that we're proud of

Turning an infeasible shape into a running one, on a *free* GPU, from a laptop with no GPU at all.

But more than the speedup: **declining the 4.014× we had already measured.** It passed every test in front of us and would have looked better on a leaderboard. It sits at $0.98\times$ of a hard gate, and shipping it would have been a coin flip dressed up as a result.

And the negative results are first-class citizens of the repository. `torch.compile` *hurts* the launch-bound shape ($2.29\times \to 1.46\times$) and is a wash on the GEMM-bound one. The stage ablation is 4 shapes $\times$ 4 configurations with every number published, including the ones that did not help.

## What we learned

- **Read the grading contract before optimizing anything.** The $\textbf{or}$ in the gate is why a reported `max_rel` of $2 \times 10^5$ can be a perfectly healthy result — it is computed as

$$\text{max\_rel} = \max_i \frac{|o_i - r_i|}{\max(|r_i|, 10^{-12})}$$

over near-zero references — and why a `max_abs` of $2.04\times10^{-3}$ is disqualifying even on a shape that passed.
- **A memory bug can surface one line away from its cause.** The OOM appeared in the attention and lived in the concat.
- **Cheap infrastructure facts dominate expensive optimization work.** One undocumented string, `NvidiaTeslaT4`, was worth more than any kernel change we made: it unlocked three levers at once.
- **Audit your own claims adversarially.** Four of ours were wrong, and all four had survived several careful re-readings by the people who wrote them.
- **Measure the compiler too.** The ablation showed `torch.compile` losing on the launch-bound shape, so the shipped model times eager, compiled, and the eager kernels captured into a CUDA graph once per shape, inside warmup, and keeps the fastest. Eager won on four shapes; shape 12 gained 15% ($1.971\times \to 2.271\times$). The median did not move, and that is the point: it stopped being a guess.
- **A faster operator is not a faster program.** Our kernel beat Inductor's fusion at the operator level and lost end to end twice: first because switching it on switched compilation off, then — after we fixed that — because a custom op is a boundary the compiler cannot fuse across. The unit you benchmark has to be the unit you ship.

## What's next

A kernel that fuses *more* than Inductor is willing to — the add, the LayerNorm, *and* the following projection's input cast in one pass — since matching the compiler's own fusion of two ops turned out not to be enough to beat it. The fused bias+GELU epilogue was scoped and dropped, since Inductor already fuses it; a Turing attention kernel with fp16 storage and fp32 accumulation is a multi-day effort we scoped and did not attempt. The padded fallback still materializes a dense $[B,1,S,S]$ bias, giving back exactly the $O(S^2)$ memory SDPA exists to avoid; the graded path never takes it, but making it memory-efficient is unfinished work rather than a solved problem.

The mixed assignment — fp16 attention, fp32 FFN and LayerNorm — turned out to be measurable in time, and it is not the middle ground it looks like: $2.953\times$ at a worst error of $1.72e-03$, a margin of $1.17\times$. Moving everything *after* the attention to fp32 recovered almost nothing, which locates the error floor inside the fp16 attention matmuls themselves. A real middle ground needs fp16 storage with fp32 accumulation *inside* the attention kernel, which the SDPA fp16 path does not expose — that is the kernel we would write next.

---8<--- 复制到这里为止 ---8<---

---

## 附：需要一起发给代填人的文件

| 文件 | 用途 | 位置 |
|---|---|---|
| `track3-submission.zip` | 第 8 项上传（4.6 MB） | `docs/`，不在 Git 里 — 用 `python scripts/make_submission_zip.py` 重新生成 |
| `01_results.png` ~ `05_demo.png` | 第 9 项画廊 | `docs/video/gallery/` |
| `track3_demo.mp4` + `.srt` | 已传 YouTube（https://youtu.be/3aAw-jq1oTM）；源文件留档 | `build/` |

## 附：如果时间不够

按这个优先级，交上去比填完整更重要：

1. Track + Project name + 仓库链接 + About 正文 → **先点提交**
2. 视频链接、zip、图片、tags → Devpost 截止前可以继续编辑补上

## 附：这些数字从哪来（被问到时的答案）

每个数字都能追到提交进仓库的原始 kernel 日志：

| 数字 | 来源 |
|---|---|
| T4 中位 2.286× / 13-13 PASS(autotune 默认,含 CUDA-graph 候选) | `results/results_t4.csv`、`results/kaggle_t4_graph_run.log` |
| T4 固定开启 compile 的对照 2.282× | `results/results_t4_compile_on.csv`、`results/kaggle_t4_run.log` |
| P100 中位 2.065× | `results/results.csv`、`results/kaggle_p100_run.log` |
| shape 14：204 s / 15,676 tok/s / 14.58 GB | `results/kaggle_t4_shape14.log` |
| fp16 中位 4.014×、`max_abs` 2.04e-3 | `results/results_t4_fp16.csv`、`results/kaggle_t4_fp16_run.log` |
| 阶段消融 4 shape × 4 stage | `results/ablation_t4.csv`、`results/kaggle_t4_ablation.log` |
| 20.5 TB | `32 × 16 × 100000² × 4 B`，图见 `report/figures/memory_wall.png` |
