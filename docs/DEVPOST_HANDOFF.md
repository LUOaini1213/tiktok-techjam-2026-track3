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
A drop-in Transformer layer that is 2.28x faster across all 13 graded shapes while staying 1049x inside the tolerance gate — and runs the 100,000-token shape whose reference needs 20.5 TB and cannot.
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
python, pytorch, cuda, triton, flash-attention, scaled-dot-product-attention, torch-compile, torch-inductor, cuda-graphs, gpu, kernel-optimization, mixed-precision, fp16, memory-optimization, transformer, attention, kaggle, tesla-t4, tesla-p100, matplotlib, pillow, edge-tts, moviepy, claude-code
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

**Devpost 的正文编辑器支持 Markdown。把下面分隔线之间的全部内容原样复制粘贴进去。**

---8<--- 从这里开始复制 ---8<---

## Inspiration

The task looked like a speed contest and turned out to be a correctness contest. The grader checks **every single output element** — `abs_err ≤ 0.002` **OR** `rel_err ≤ 0.02` — and one failing element fails the whole shape and forfeits the speed score entirely. So the interesting question was never "how fast can this go", it was "how fast can this go while the answer provably does not move".

Then we read the shape list. Shape 14 asks for `seq_len = 100,000`, and the reference implementation materializes its attention scores explicitly: `[32, 16, 100000, 100000]` = 5.12×10¹² elements ≈ **20.5 TB** in fp32. No GPU holds that. The baseline does not run slowly on that shape — it does not run. That is the shape we wanted.

## What it does

**ExactSwap** is a drop-in replacement for the reference `BaselineTransformer`. It subclasses it, keeps every submodule and parameter name, and rewrites only the forward compute — so the official `copy_model_weights(..., strict=True)` succeeds and the comparison is apples to apples with zero friction.

**On a free Kaggle Tesla T4, fp32 grading: 13/13 gradeable shapes PASS, median speedup 2.282× (1.086×–4.439×), worst absolute error 1.91e-6 — about 1049× inside the tolerance gate.**

**And shape 14 runs.** The full 100,000-token forward completes in **204 s at 15,676 tokens/s, peaking at 14.58 GB** — on a card that only has 15.64 GB in total. Against a reference that needs 20.5 TB and cannot execute at all, the meaningful result is not a ratio; it is the difference between *cannot run* and *runs*.

Three levers, all selectable by environment variable with no code edits, so the ablation and the delivered path are literally the same code:

1. **`F.scaled_dot_product_attention`** — `O(S)` memory instead of the baseline's `O(S²)`. The score matrix is never built. Causality goes through `is_causal=True`, generated inside the kernel, because a dense `[S,S]` mask at `S=100,000` would be 10 GB on its own.
2. **Self-applied `torch.compile`** — the model compiles itself on first forward, so the speedup does not depend on the grader passing `--compile-user`. Needs `sm ≥ 7.0`.
3. **VRAM-planned batch chunking** — the chunk size is computed from the memory *actually free at runtime* (`cuda.mem_get_info` plus the allocator's reserved-but-unused blocks) minus the output buffer, and halves and retries if that estimate was optimistic.

## How we built it

Development happened on a laptop with an Intel iGPU and **no NVIDIA card**. Every GPU run was pushed headlessly to Kaggle and pulled back: a builder inlines the unmodified harness, the optimized model and a sweep driver into one self-contained kernel, so nothing needs to be attached to the notebook.

**The three results that were not obvious going in:**

**1. The shape-14 OOM was not in the attention.** After switching to fused attention, shape 14 still died with `Tried to allocate 6.10 GiB`. The fused kernel was fine. The failure was the last line: batch-chunking the forward and joining the pieces with `torch.cat` holds the chunks **and** the joined `[32, 100000, 1024]` result at the same time — a second ~6.5 GB allocation on a card that already had the 6.5 GB input resident. Writing each chunk straight into a preallocated output removed the copy, and the shape ran. The bug was one line away from where it appeared.

**2. The Kaggle API silently gives you the wrong GPU.** `torch.compile`, real FlashAttention, and fp16 tensor cores were all listed as "not measured" for most of this project, because the API-allocated card is a Tesla **P100** (`sm_60`) where Triton will not build. The kernels API *does* let you choose — `machine_shape` / `--accelerator` — but the accepted values (`NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`) appear only in the SDK docstring for `ApiSaveKernelRequest`, and **anything unrecognised is silently normalised back to a P100**. Our first two guesses looked like successful requests and were not. We confirmed the right value by pushing a throwaway kernel that just printed `get_device_name`.

**3. Two GPUs make an accidental ablation.** The same code scores 2.065× on the P100 (SDPA alone) and 2.282× on the T4 (SDPA + compile + FlashAttention). Worth saying out loud: the T4's *baselines* are **slower** than the P100's — 318.7 ms vs 168.6 ms on shape 13 — because the P100 has more fp32 throughput and roughly twice the memory bandwidth. Our ratio improves on the T4 anyway, because the optimized path picks up compilation and FlashAttention there while the baseline stays bandwidth-bound. A speedup is a ratio, and it matters which side moved.

## Challenges we ran into

**We shipped a claim we had never measured, and measuring it proved us wrong.** The repository asserted for most of its life that internal fp16 "breaks the strict `atol=0.002` gate". When we finally got a T4 and ran it, fp16 **passed all 13 shapes**, at a median **4.014×** — nearly double the shipped path. The assertion was false.

The conclusion survived anyway, for a better reason. fp16's worst absolute error is `max_abs = 0.0020388` on shape 6 — it has **already crossed** `atol = 0.002`. It passes only because the gate is `abs ≤ 0.002` **OR** `rel ≤ 0.02`, and that particular element happened to have a reference value large enough (`|ref| ≥ 0.102`) for the relative branch to catch it. Nothing makes that repeatable: put the same error on a near-zero reference and the element fails, and one failing element forfeits the speed score. So the trade is roughly **2× more speed for a correctness margin of essentially zero** on a hard gate, against fp32's 1049×. We took the margin, and left fp16 as a documented, measured flag. **The conclusion was right and the reasoning behind it was wrong, and only running it revealed which.**

**We ran an adversarial audit against our own documentation, and it found four wrong claims.** Multiple agents cross-checked every number in the README, report, ablation and Devpost draft against `results/*.csv` and the raw kernel logs, and every finding was independently verified before being accepted. Confirmed and fixed: shape 14's headline numbers were fp16 while every document framed the run as fp32; the Devpost draft credited speedups to "tensor cores + compilation" on a run where the GPU had neither; a "~2000× inside tolerance" figure was arithmetic on a number no shape actually hit (the real margin is ~1049×); and the README pointed at a `kernels/` directory that never existed. The audit also found a real bug in our own tooling — the log parser raised `NameError` on any log containing a shape-level error — and, with some irony, one audit agent triggered the exact hazard it had just reported by running `run_all.py` on this CPU-only machine and overwriting the committed GPU results. Nothing reached the remote, the file regenerates from the committed log exactly, and the script now refuses that write.

**Two findings turned out not to be defects, and proving that mattered.** `--dtype bfloat16` fails the accuracy gate: 6131 of 131072 elements, `max_abs 0.047`. So does **the reference compared against itself** recomputed in fp32 — 7603 elements, identical `max_abs`. bf16's ulp near 1.0 is 0.0078, four times coarser than `atol=0.002`, so no implementation that reorders a single operation can hold that gate in bf16. It is a property of that configuration, not of our kernel; the graded configuration is fp32.

## Accomplishments that we're proud of

Turning an infeasible shape into a running one, on a *free* GPU, from a laptop with no GPU at all.

But more than the speedup: **declining the 4.01× we had already measured.** It passed every test in front of us. It would have looked better on a leaderboard. It sits at 0.98× of a hard gate, and shipping it would have been a coin flip dressed as a result.

And the negative results are first-class citizens of the repository. `torch.compile` *hurts* the launch-bound shape (2.29× → 1.46×) and is a wash on the GEMM-bound one. The stage ablation is 4 shapes × 4 configurations with every number published, including the ones that did not help.

## What we learned

- **Read the grading contract before optimizing.** The `abs OR rel` structure is why a `max_rel` of 200,000 can be a completely healthy result — it is computed over near-zero references — and why a `max_abs` of 0.00204 is disqualifying even though the shape passed.
- **A memory bug can surface one line away from its cause.** The OOM appeared in the attention and lived in the concat.
- **Cheap infrastructure facts dominate expensive optimization work.** One undocumented string, `NvidiaTeslaT4`, was worth more than any kernel change we made — it unlocked three levers at once.
- **Audit your own claims adversarially.** Four of ours were wrong, and all four had survived several careful re-readings by the people who wrote them.

## What's next

Hand-written Triton kernels — a fused LayerNorm+residual and a fused bias+GELU epilogue — beyond what Inductor already fuses; a Turing-specific FlashAttention kernel is a multi-day effort we scoped and did not attempt. The padded fallback still materializes a dense `[B,1,S,S]` bias, giving back exactly the memory SDPA exists to avoid; the graded path never takes it, but making it memory-efficient is unfinished work rather than a solved problem. And the obvious experiment we ran out of time for: a mixed assignment — fp16 matmuls with selected fp32 stages — that keeps most of the 2× while restoring real tolerance margin.

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
| T4 中位 2.282× / 13-13 PASS | `results/results_t4.csv`、`results/kaggle_t4_run.log` |
| P100 中位 2.065× | `results/results.csv`、`results/kaggle_p100_run.log` |
| shape 14：204 s / 15,676 tok/s / 14.58 GB | `results/kaggle_t4_shape14.log` |
| fp16 中位 4.014×、`max_abs` 2.04e-3 | `results/results_t4_fp16.csv`、`results/kaggle_t4_fp16_run.log` |
| 阶段消融 4 shape × 4 stage | `results/ablation_t4.csv`、`results/kaggle_t4_ablation.log` |
| 20.5 TB | `32 × 16 × 100000² × 4 B`，图见 `report/figures/memory_wall.png` |
