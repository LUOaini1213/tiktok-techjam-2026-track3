#!/usr/bin/env python3
"""
Zero-friction submission entry point.

This runs the OFFICIAL benchmark unchanged, but swaps in our optimized model.
The official ``main()`` looks up ``UserOptimizedTransformer`` as a module global,
so replacing that attribute before calling ``main()`` makes the harness build
and grade our implementation with exactly the same CLI, timing and correctness
logic as the original ``torch_transformer_benchmark.py``.

Usage is identical to the official script, e.g.:

    python submission.py --causal --device cuda --dtype float32 \
        --batch-size 64 --d-model 128 --heads 4 --seq-len 128 \
        --layers 4 --ffn-dim 128
"""

from __future__ import annotations

import torch_transformer_benchmark as bench
from user_optimized import UserOptimizedTransformer

# Swap the reference stub for our optimized model (same weights, faster forward).
bench.UserOptimizedTransformer = UserOptimizedTransformer

if __name__ == "__main__":
    raise SystemExit(bench.main())
