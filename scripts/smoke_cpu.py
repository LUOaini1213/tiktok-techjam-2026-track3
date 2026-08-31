#!/usr/bin/env python3
"""
Fast CPU correctness smoke test (no GPU, no timing).

Validates the CORE math of UserOptimizedTransformer against the reference on
small/medium shapes, in fp32 on CPU. On CPU our model runs the eager SDPA path
(autocast + torch.compile are CUDA-only and skipped), so this checks reshape /
head-split / causal / mask / weight-name-compat / output shape+dtype — exactly
the parts that must be correct before spending cloud-GPU quota.

The fp16 and torch.compile paths only activate on CUDA and are validated on Colab.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_transformer_benchmark as bench
from user_optimized import UserOptimizedTransformer

# [B, d_model, heads, seq, layers, ffn] — skip 6 (B=10000) and 14 (infeasible) on CPU.
CASES = [
    (1, 64, 128, 4, 128, 4, 128),
    (2, 1, 128, 4, 128, 4, 128),
    (3, 4, 128, 4, 128, 4, 128),
    (4, 16, 128, 4, 128, 4, 128),
    (5, 128, 128, 4, 128, 4, 128),
    (7, 64, 32, 4, 128, 4, 32),
    (8, 64, 1024, 4, 128, 4, 1024),
    (9, 64, 128, 1, 128, 4, 128),
    (10, 64, 128, 2, 128, 4, 128),
    (11, 64, 128, 16, 128, 4, 128),
    (12, 64, 128, 4, 32, 4, 128),
    (13, 64, 128, 4, 1024, 4, 128),
]

RTOL, ATOL = 0.02, 0.002


def check(idx, b, d, h, s, l, f, padding_ratio=0.0, trials=2):
    cfg = bench.TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                  ffn_dim=f, num_layers=l, causal=True)
    cfg.validate()
    baseline = bench.BaselineTransformer(cfg)
    optimized = UserOptimizedTransformer(cfg)
    bench.copy_model_weights(baseline, optimized, strict=True)  # strict name check
    baseline = baseline.to("cpu", torch.float32).eval()
    optimized = optimized.to("cpu", torch.float32).eval()

    ok = True
    mabs = mrel = 0.0
    with torch.inference_mode():
        for t in range(trials):
            x, mask = bench.generate_random_case(cfg, torch.device("cpu"),
                                                 torch.float32, seed=1234 + t,
                                                 padding_ratio=padding_ratio,
                                                 input_scale=1.0)
            ref = baseline(x, mask)
            opt = optimized(x, mask)
            assert opt.shape == ref.shape, (opt.shape, ref.shape)
            assert opt.dtype == torch.float32
            assert torch.isfinite(opt).all(), "non-finite output!"
            res = bench.compare_outputs(ref, opt, rtol=RTOL, atol=ATOL)
            ok &= res.passed
            mabs = max(mabs, res.max_abs_error)
            mrel = max(mrel, res.max_relative_error)
    tag = f"pad={padding_ratio}" if padding_ratio else ""
    print(f"shape {idx:>2} B={b:<5} D={d:<4} H={h:<2} S={s:<6} F={f:<4} "
          f"{'PASS' if ok else 'FAIL':4} max_abs={mabs:.2e} max_rel={mrel:.2e} {tag}")
    return ok


def main() -> int:
    print(f"torch {torch.__version__} | device cpu | tol atol={ATOL} rtol={RTOL}\n")
    all_ok = True
    for c in CASES:
        all_ok &= check(*c)
    # Exercise the padded fallback (mask + nan_to_num) on a small shape.
    print("\n-- padded fallback --")
    all_ok &= check(2, 1, 128, 4, 128, 4, 128, padding_ratio=0.3)
    all_ok &= check(4, 16, 128, 4, 128, 4, 128, padding_ratio=0.5)
    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
