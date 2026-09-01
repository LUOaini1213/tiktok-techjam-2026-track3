# 3-minute demo video script — Track 3

> Public YouTube, ≤ 3:00. Screen-recording + voiceover. Replace [NUM] after the run.
> A walkthrough (no fancy UI) is explicitly accepted for backend tracks.

**0:00–0:20 — Hook / problem.**
"Track 3: make a Transformer layer faster on the GPU, but bit-for-bit correct.
Here's the reference — standard multi-head attention plus FFN — and the 14 test
shapes it's graded on." (Show `torch_transformer_benchmark.py` baseline + the
shapes appendix.)

**0:20–0:50 — The memory wall (the insight).**
"Shape 14 has sequence length 100,000. The baseline builds an S×S attention
matrix — that's 32·16·100000·100000, about 20.5 **terabytes**. No GPU can hold
it. So the baseline literally cannot run this shape." (Show `memory_wall.png`.)

**0:50–1:25 — Our fix runs it.**
"We reformulate attention with scaled_dot_product_attention — FlashAttention —
which never materializes that matrix. Same math, O(S) memory. On a free cloud
GPU, shape 14 runs in [MS] ms, [TOK/S] tokens/second, under [GB] GB." (Show the
shape-14 log: correctness at truncated length, then full-length timing.)

**1:25–2:05 — Speed across the board.**
"For the other shapes the win is the same fused attention kernel, plus
self-applied torch.compile where the GPU supports it. We also built an fp16
tensor-core path — but the grader's absolute tolerance is unforgiving for
near-zero outputs, so we ship it off by default and stay exact. Here's the full
sweep." (Show `run_all` / results table + `speedups.png`: median [MEDIAN]×,
all PASS, max_abs about 1e-6.)

**2:05–2:35 — How it stays correct & drop-in.**
"We keep every baseline parameter name and rewrite only the forward, so the
official strict weight-copy and comparison run unchanged. Softmax stays fp32,
GELU is exact, fully-masked rows are guarded — zero failed elements." (Show the
`user_optimized.py` forward + a PASS summary.)

**2:35–3:00 — Free cloud + AI, close.**
"All of this ran on free Colab/Kaggle GPUs from a laptop with no NVIDIA card,
driven headlessly. We used Claude to analyze the harness, design the approach,
and write the kernels — logged in AI_TOOLS.md. From impossible to [MEDIAN]×
faster." (Show repo + headline number.)

## Capture checklist
- [ ] baseline shape-14 OOM (or the 20.5 TB math on screen)
- [ ] optimized shape-14 running (log: tokens/s, VRAM)
- [ ] results table with all PASS + speedups
- [ ] `memory_wall.png`, `speedups.png`
- [ ] 10s of `user_optimized.py` forward
- [ ] repo + AI_TOOLS.md
