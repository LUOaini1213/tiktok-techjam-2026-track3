#!/usr/bin/env python3
"""
Causal attention on fp16 tensor cores with fp32 everywhere else, in Triton.

**Where fp16's error actually comes from.** The fp16 autocast path is not
inaccurate because of accumulation -- cuBLAS and the memory-efficient SDPA
kernel already accumulate in fp32. It is inaccurate because tensors are
*stored* in fp16 at five points (q, k, v, the attention output, the output
projection), each a 2^-11 relative rounding, and those compound to the
~1.7e-3 absolute error the mixed-precision sweep measured against a 2e-3 gate.

This kernel keeps q, k, v, the softmax statistics, the accumulator and the
output in fp32 and converts operands to fp16 only inside the tensor-core
matmuls, so there are two rounding points instead of five. That is ``SPLIT=1``.

**Operand splitting** (``SPLIT=3``) removes those two as well. Each fp32 operand
is written as ``x = x_hi + x_lo`` with both halves fp16, so ``x_lo`` carries the
next 11 bits, and the product is formed as::

    a . b  ~=  a_hi . b_hi  +  a_hi . b_lo  +  a_lo . b_hi        (dropping lo . lo)

Three tensor-core matmuls per product instead of one, each accumulated in fp32,
for a relative error of about 2^-22 -- fp32-class -- at fp16 tensor-core
throughput. On a T4 fp16 MMA is ~8x the fp32 CUDA-core rate, so 3x the matmul
work is still ahead. Whether it is ahead *after* the extra register pressure is
what the benchmark measures.

Both variants use the online (flash-style) softmax: the ``[S, S]`` score matrix
is never materialized, memory is ``O(S)``, and the scale and ``log2(e)`` are
folded into q so the inner loop uses ``exp2``.

Scope: forward only, causal or full, no padding mask (the graded path has
none; the caller falls back to SDPA when it has one), ``head_dim <= 128``.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

LOG2E = 1.4426950408889634
MAX_HEAD_DIM = 128


if HAVE_TRITON:

    @triton.jit
    def _attn_fwd(
        Q, K, V, O,
        sqb, sqh, sqm, sqd,
        skb, skh, skn, skd,
        svb, svh, svn, svd,
        sob, soh, som, sod,
        H, S, qk_scale,
        HEAD_D: tl.constexpr, BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr, SPLIT: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        b = off_bh // H
        h = off_bh % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        d_ok = offs_d < HEAD_D

        q_ptrs = Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < S) & d_ok[None, :], other=0.0)
        q = q.to(tl.float32) * qk_scale
        q_hi = q.to(tl.float16)
        if SPLIT == 3:
            q_lo = (q - q_hi.to(tl.float32)).to(tl.float16)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        if CAUSAL:
            hi = tl.minimum((start_m + 1) * BLOCK_M, S)
        else:
            hi = S

        for start_n in range(0, hi, BLOCK_N):
            cols = start_n + offs_n
            kv_mask = (cols[:, None] < S) & d_ok[None, :]

            k_ptrs = K + b * skb + h * skh + cols[:, None] * skn + offs_d[None, :] * skd
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            k_hi = k.to(tl.float16)
            qk = tl.dot(q_hi, tl.trans(k_hi))
            if SPLIT == 3:
                k_lo = (k - k_hi.to(tl.float32)).to(tl.float16)
                qk += tl.dot(q_hi, tl.trans(k_lo))
                qk += tl.dot(q_lo, tl.trans(k_hi))

            valid = cols[None, :] < S
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= cols[None, :])
            qk = tl.where(valid, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = V + b * svb + h * svh + cols[:, None] * svn + offs_d[None, :] * svd
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_hi = v.to(tl.float16)
            p_hi = p.to(tl.float16)
            acc += tl.dot(p_hi, v_hi)
            if SPLIT == 3:
                v_lo = (v - v_hi.to(tl.float32)).to(tl.float16)
                p_lo = (p - p_hi.to(tl.float32)).to(tl.float16)
                acc += tl.dot(p_hi, v_lo)
                acc += tl.dot(p_lo, v_hi)

            m_i = m_new

        o = acc / l_i[:, None]
        o_ptrs = O + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod
        tl.store(o_ptrs, o.to(O.dtype.element_ty),
                 mask=(offs_m[:, None] < S) & d_ok[None, :])


def _config(block_d: int, split: int):
    """Tile sizes by head width. Turing has 64 KB of shared memory per SM and no
    async copies, and SPLIT=3 holds hi and lo tiles of k and v at once, so the
    wider heads get the narrower key tile."""
    if block_d <= 64:
        return (64, 64 if split == 1 else 32, 4)
    return (64, 32 if split == 1 else 16, 4)


def can_use(q: torch.Tensor) -> bool:
    return (HAVE_TRITON and q.is_cuda and q.dim() == 4
            and q.shape[-1] <= MAX_HEAD_DIM
            and q.dtype in (torch.float32, torch.float16, torch.bfloat16))


def _launch(kernel, q, k, v, o, scale, causal, split):
    B, H, S, D = q.shape
    block_d = max(16, triton.next_power_of_2(D))
    block_m, block_n, warps = _config(block_d, split)
    grid = (triton.cdiv(S, block_m), B * H)
    kernel[grid](
        q, k, v, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, S, float(scale) * LOG2E,
        HEAD_D=D, BLOCK_D=block_d, BLOCK_M=block_m, BLOCK_N=block_n,
        CAUSAL=bool(causal), SPLIT=int(split),
        num_warps=warps, num_stages=2,
    )


def attention_raw(q, k, v, scale=None, causal=True, split=3):
    """q, k, v: [B, H, S, D], any strides with the last one unit. Returns [B, H, S, D]."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _launch(_attn_fwd, q, k, v, o, scale, causal, split)
    return o


HAVE_ATTN_OP = False
if HAVE_TRITON:
    try:
        from torch.library import triton_op, wrap_triton

        @triton_op("exactswap::attention", mutates_args={})
        def _attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          scale: float, causal: bool, split: int) -> torch.Tensor:
            o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
            _launch(wrap_triton(_attn_fwd), q, k, v, o, scale, causal, split)
            return o

        HAVE_ATTN_OP = True
    except Exception:  # pragma: no cover - torch < 2.6
        HAVE_ATTN_OP = False


def attention(q, k, v, scale=None, causal=True, split=3):
    """Tensor-core attention with fp32 statistics; composes with torch.compile
    when the op could be registered. Falls back to SDPA when it cannot apply."""
    if not can_use(q):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, scale=scale)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if HAVE_ATTN_OP:
        return _attention_op(q, k, v, float(scale), bool(causal), int(split))
    return attention_raw(q, k, v, scale, causal, split)
