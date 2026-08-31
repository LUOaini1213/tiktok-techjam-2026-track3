#!/usr/bin/env python3
"""
Shape 14 (batch=32, d_model=1024, heads=16, seq_len=100000, layers=2, ffn=1024).

The baseline cannot run this: its explicit attention needs a [B,H,S,S] score
matrix = 32*16*1e5*1e5 = 5.12e12 elements = ~20.5 TB in fp32. No GPU can hold
it, so the standard harness (which runs baseline first) dies before timing.

This script demonstrates the value of the optimization:
  A) TIMING: run ONLY the optimized model at the full seq_len=100000 (fp16,
     batch-chunked, FlashAttention via SDPA) and report latency + tokens/s.
  B) CORRECTNESS-BY-CONSTRUCTION: at a truncated seq_len the baseline CAN run,
     compare optimized vs baseline element-wise with the official tolerances.
     SDPA attention is mathematically identical regardless of S, so passing at
     the truncated length evidences correctness at S=100000.

Run on a 16 GB GPU (Kaggle T4/P100 recommended). Example:
    python scripts/shape14_optimized_only.py --trunc-seq 2048
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                    # flat layout (e.g. Kaggle working dir)
sys.path.insert(0, os.path.dirname(_here))   # repo layout (scripts/ under repo root)

import torch_transformer_benchmark as bench
from user_optimized import UserOptimizedTransformer

FULL = dict(batch_size=32, seq_len=100000, d_model=1024,
            num_heads=16, ffn_dim=1024, num_layers=2, causal=True)


def build_pair(cfg, device, dtype):
    baseline = bench.BaselineTransformer(cfg)
    optimized = UserOptimizedTransformer(cfg)
    bench.copy_model_weights(baseline, optimized, strict=True)
    return (baseline.to(device=device, dtype=dtype).eval(),
            optimized.to(device=device, dtype=dtype).eval())


def timing_full(args, device):
    print("\n=== A) full seq_len=100000, optimized only (fp16) ===")
    cfg = bench.TransformerConfig(**{**FULL, "num_layers": args.layers})
    dtype = torch.float16
    try:
        _, optimized = build_pair(cfg, device, dtype)
        x, mask = bench.generate_random_case(cfg, device, dtype, seed=1234,
                                             padding_ratio=0.0, input_scale=1.0)
        with torch.inference_mode():
            for _ in range(args.warmup):
                optimized(x, mask)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(args.iters):
                start.record()
                optimized(x, mask)
                end.record()
                torch.cuda.synchronize(device)
                samples.append(start.elapsed_time(end))
        samples.sort()
        med = samples[len(samples) // 2]
        tokens = cfg.batch_size * cfg.seq_len
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  median={med:.2f} ms | tokens/call={tokens} | "
              f"throughput={tokens*1000.0/med:,.0f} tok/s | peak_vram={peak:.2f} GB")
        print(f"  chunk_bs={optimized._chunk_bs} autocast={optimized._autocast_dtype}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [OOM] full seq_len did not fit on this GPU: {e}")
            print("  -> try a 16GB+ GPU (Kaggle P100/T4), or set T3_CHUNK_BS=1.")
        else:
            raise


def correctness_truncated(args, device):
    print(f"\n=== B) truncated seq_len={args.trunc_seq}, correctness vs baseline ===")
    cfg = bench.TransformerConfig(**{**FULL, "seq_len": args.trunc_seq,
                                     "num_layers": args.layers})
    dtype = torch.float32  # match grading semantics for the reference
    baseline, optimized = build_pair(cfg, device, dtype)
    x, mask = bench.generate_random_case(cfg, device, dtype, seed=1234,
                                         padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        ref = baseline(x, mask)
        opt = optimized(x, mask)
    res = bench.compare_outputs(ref, opt, rtol=0.02, atol=0.002)
    print(f"  {'PASS' if res.passed else 'FAIL'} | max_abs={res.max_abs_error:.6g} | "
          f"max_rel={res.max_relative_error:.6g} | failed={res.failed_elements}/"
          f"{res.total_elements}")
    return res.passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunc-seq", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required. Run on Colab/Kaggle GPU.")
        return 1
    device = torch.device("cuda")
    print(f"gpu={torch.cuda.get_device_name(device)} torch={torch.__version__}")

    ok = correctness_truncated(args, device)
    if not args.skip_full:
        timing_full(args, device)
    print("\nSummary: correctness(truncated)=" + ("PASS" if ok else "FAIL")
          + "; full-seq timing above (baseline is infeasible by construction).")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
