# Ablation study

Incremental contribution of each optimization, measured on the same GPU with
the official harness. Fill in after the cloud runs. Speedup is vs the baseline
(the harness reports `speedup = baseline.median / optimized.median`).

Toggle each stage with environment variables (no code edits):

| Stage | Command (shape 1 example) |
|---|---|
| SDPA only (fp32, no compile) | `T3_AUTOCAST=off T3_COMPILE=0 python run_all.py --shapes 1` |
| + fp16 autocast | `T3_COMPILE=0 python run_all.py --shapes 1` |
| + torch.compile | `python run_all.py --shapes 1` |
| + shape-14 chunking | `python scripts/shape14_optimized_only.py` |

## Representative shapes

### Shape 1 — [B=64, D=128, H=4, S=128, L=4, F=128] (compute)
| stage | median opt (ms) | speedup | max_rel | pass |
|---|---|---|---|---|
| SDPA only | | | | |
| + fp16 | | | | |
| + compile | | | | |

### Shape 6 — [B=10000, …] (throughput)
| stage | median opt (ms) | speedup | max_rel | pass |
|---|---|---|---|---|
| SDPA only | | | | |
| + fp16 | | | | |
| + compile | | | | |

### Shape 13 — [S=1024] (long-sequence / memory)
| stage | median opt (ms) | speedup | max_rel | pass |
|---|---|---|---|---|
| SDPA only | | | | |
| + fp16 | | | | |
| + compile | | | | |

### Shape 14 — [S=100000] (extreme)
Baseline infeasible (20.5 TB scores). Report optimized latency + tokens/s and
peak VRAM from `scripts/shape14_optimized_only.py`; correctness by construction
at truncated `seq_len`.

| metric | value |
|---|---|
| full-seq median (ms) | |
| throughput (tok/s) | |
| peak VRAM (GB) | |
| truncated-S correctness | |
