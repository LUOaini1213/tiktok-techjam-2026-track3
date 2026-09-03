# 3-minute demo video script — Track 3

> **Note.** An earlier draft of this storyboard called the attention kernel "FlashAttention". A
> backend probe later showed neither GPU could run PyTorch's flash kernel (fp16-only, sm_80+);
> every run used SDPA's memory-efficient backend. The line below is corrected. The uploaded
> video is unaffected: its narration (`build/track3_demo.srt`, generated from
> `scripts/make_video.py`) says "fused attention" and never used the word.


> **The delivered video is generated, not hand-edited.** `scripts/make_video.py`
> renders the slides from `results/*.csv`, synthesizes the narration, and holds
> each slide for exactly as long as its own voice-over runs — so re-running it
> after a new sweep produces a video that still agrees with the data:
>
> ```bash
> python scripts/make_video.py --out build/track3_demo.mp4
> ```
>
> Output: `build/track3_demo.mp4` (2:22, 1920x1080) plus `.srt` subtitles.
> The narration text lives in that script's `SCENES` list; this document is the
> storyboard it was written from, kept for the reasoning behind each beat.

> Public YouTube, ≤ 3:00. Screen-recording + voiceover. Numbers are the measured
> Kaggle P100 run (results/kaggle_p100_run.log).
> A walkthrough (no fancy UI) is explicitly accepted for backend tracks.

**0:00–0:20 — Hook / problem.**
"Track 3: make a Transformer layer faster on the GPU, without moving the answer.
Here's the reference — standard multi-head attention plus FFN — and the 14 test
shapes it's graded on." (Show `torch_transformer_benchmark.py` baseline + the
shapes appendix.)

**0:20–0:50 — The memory wall (the insight).**
"Shape 14 has sequence length 100,000. The baseline builds an S×S attention
matrix — that's 32·16·100000·100000, about 20.5 **terabytes**. No GPU can hold
it. So the baseline literally cannot run this shape." (Show `memory_wall.png`.)

**0:50–1:25 — Our fix runs it.**
"We reformulate attention with scaled_dot_product_attention — the memory-efficient fused kernel —
which never materializes that matrix. Same math, O(S) memory. On a free cloud
GPU — a 16 GB P100 — shape 14 runs: 293 seconds per forward, about 11,000 tokens
a second, peaking at 14.6 gigabytes. That run is fp16, and for this shape that
isn't a shortcut — in fp32 the input and the output alone are 26 gigabytes."
(Show the shape-14 log: the fp32 correctness check at truncated length, then the
full-length timing line.)

**1:25–2:05 — Speed across the board.**
"For the other shapes the win is the same fused attention kernel, plus
self-applied torch.compile where the GPU supports it. We also built an fp16
tensor-core path that is nearly twice as fast again — and we ship it turned off.
It passes every shape, but its worst error has already crossed the absolute
tolerance and only survives on the relative one. That is luck, not margin. Here's the full
sweep." (Show `run_all` / results table + `speedups.png`: median 2.28x on a T4 (2.07x on a P100),
all PASS, max_abs about 1e-6.)

**2:05–2:35 — How it stays correct & drop-in.**
"We keep every baseline parameter name and rewrite only the forward, so the
official strict weight-copy and comparison run unchanged. Softmax stays fp32,
GELU is exact, fully-masked rows are guarded — zero failed elements." (Show the
`user_optimized.py` forward + a PASS summary.)

**2:35–3:00 — Free cloud + AI, close.**
"All of this ran on free Colab/Kaggle GPUs from a laptop with no NVIDIA card,
driven headlessly. We used Claude to analyze the harness, design the approach,
and write the kernels — logged in AI_TOOLS.md. From impossible to 2.28x
faster." (Show repo + headline number.)

## Capture checklist
- [ ] baseline shape-14 OOM (or the 20.5 TB math on screen)
- [ ] optimized shape-14 running (log: tokens/s, VRAM)
- [ ] results table with all PASS + speedups
- [ ] `memory_wall.png`, `speedups.png`
- [ ] 10s of `user_optimized.py` forward
- [ ] repo + AI_TOOLS.md
