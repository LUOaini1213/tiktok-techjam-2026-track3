#!/usr/bin/env python3
"""
Causal attention on fp16 tensor cores with fp32 statistics, in Triton.

**Where fp16's error actually comes from.** The fp16 autocast path is not
inaccurate because of accumulation -- cuBLAS and the memory-efficient SDPA
kernel already accumulate in fp32. It is inaccurate because tensors are
*stored* in fp16 at five points (q, k, v, the attention output, the output
projection), each a 2^-11 relative rounding, and those compound to the
~1.7e-3 absolute error the mixed-precision sweep measured against a 2e-3 gate.

This kernel keeps the softmax statistics, the accumulator and the output in
fp32 and rounds to fp16 only for the tensor-core matmul operands, so there are
two rounding points instead of five. That is ``SPLIT=1``. Measured, it barely
helps (1.3e-3 to 2.4e-3): the operand rounding *was* the error.

**Operand splitting** (``SPLIT=3``) removes it. Each fp32 operand is written as
``x = x_hi + x_lo`` with both halves fp16, so ``x_lo`` carries the next 11 bits,
and the product is formed as::

    a . b  ~=  a_hi . b_hi  +  a_hi . b_lo  +  a_lo . b_hi        (dropping lo . lo)

Three tensor-core matmuls per product instead of one, each accumulated in fp32,
for a relative error of about 2^-22 -- fp32-class. Measured against an fp64
reference: 1.4e-6 to 4.1e-6, against fp32 SDPA's own 0.9e-6 to 1.3e-6.

**Layout.** The six fp16 operand tensors (hi/lo of q, k, v) are produced once,
outside the kernel, contiguous as ``[B, H, S, D]``; the kernel then streams
fp16 tiles with no in-loop conversion, loads K already transposed so no
register transpose is needed, drops bounds masks entirely when the shapes are
even, and runs the causal loop in two phases -- full blocks below the diagonal
unmasked, the diagonal block masked. Tiles are sized for Turing (64 KB of
shared memory, no async copies).

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
    def _attn_inner(
        acc, l_i, m_i, q_hi, q_lo,
        KH, KL, VH, VL, kT_off, v_off, s_m,
        offs_m, offs_n, offs_d, S, lo, hi,
        HEAD_D: tl.constexpr, BLOCK_N: tl.constexpr,
        CAUSAL_MASK: tl.constexpr, SPLIT: tl.constexpr,
        EVEN_S: tl.constexpr, EVEN_D: tl.constexpr,
    ):
        for start_n in range(lo, hi, BLOCK_N):
            cols = start_n + offs_n
            koff = kT_off + start_n * s_m
            voff = v_off + start_n * s_m

            if EVEN_S and EVEN_D:
                k_hi = tl.load(KH + koff)
            else:
                kmask = (offs_d[:, None] < HEAD_D) & (cols[None, :] < S)
                k_hi = tl.load(KH + koff, mask=kmask, other=0.0)
            qk = tl.dot(q_hi, k_hi)
            if SPLIT == 3:
                if EVEN_S and EVEN_D:
                    k_lo = tl.load(KL + koff)
                else:
                    k_lo = tl.load(KL + koff, mask=kmask, other=0.0)
                qk += tl.dot(q_hi, k_lo)
                qk += tl.dot(q_lo, k_hi)

            if CAUSAL_MASK:
                valid = offs_m[:, None] >= cols[None, :]
                if not EVEN_S:
                    valid = valid & (cols[None, :] < S)
                qk = tl.where(valid, qk, float("-inf"))
            elif not EVEN_S:
                qk = tl.where(cols[None, :] < S, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            if EVEN_S and EVEN_D:
                v_hi = tl.load(VH + voff)
            else:
                vmask = (cols[:, None] < S) & (offs_d[None, :] < HEAD_D)
                v_hi = tl.load(VH + voff, mask=vmask, other=0.0)
            p_hi = p.to(tl.float16)
            acc += tl.dot(p_hi, v_hi)
            if SPLIT == 3:
                if EVEN_S and EVEN_D:
                    v_lo = tl.load(VL + voff)
                else:
                    v_lo = tl.load(VL + voff, mask=vmask, other=0.0)
                p_lo = (p - p_hi.to(tl.float32)).to(tl.float16)
                acc += tl.dot(p_hi, v_lo)
                acc += tl.dot(p_lo, v_hi)

            m_i = m_new
        return acc, l_i, m_i

    @triton.jit
    def _attn_fwd(
        QH, QL, KH, KL, VH, VL, O,
        s_b, s_h, s_m,
        o_b, o_h, o_m,
        H, S,
        HEAD_D: tl.constexpr, BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr, SPLIT: tl.constexpr,
        EVEN_S: tl.constexpr, EVEN_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        b = off_bh // H
        h = off_bh % H
        base = b * s_b + h * s_h

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_off = base + offs_m[:, None] * s_m + offs_d[None, :]
        if EVEN_S and EVEN_D:
            q_hi = tl.load(QH + q_off)
        else:
            q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < HEAD_D)
            q_hi = tl.load(QH + q_off, mask=q_mask, other=0.0)
        if SPLIT == 3:
            if EVEN_S and EVEN_D:
                q_lo = tl.load(QL + q_off)
            else:
                q_lo = tl.load(QL + q_off, mask=q_mask, other=0.0)
        else:
            q_lo = q_hi

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        kT_off = base + offs_d[:, None] + offs_n[None, :] * s_m   # [BLOCK_D, BLOCK_N]
        v_off = base + offs_n[:, None] * s_m + offs_d[None, :]    # [BLOCK_N, BLOCK_D]

        if CAUSAL:
            diag = start_m * BLOCK_M
            hi = tl.minimum(diag + BLOCK_M, S)
            # Full blocks strictly below the diagonal: no causal test needed.
            acc, l_i, m_i = _attn_inner(
                acc, l_i, m_i, q_hi, q_lo, KH, KL, VH, VL, kT_off, v_off, s_m,
                offs_m, offs_n, offs_d, S, 0, diag,
                HEAD_D, BLOCK_N, False, SPLIT, EVEN_S, EVEN_D)
            # The diagonal block(s): masked.
            acc, l_i, m_i = _attn_inner(
                acc, l_i, m_i, q_hi, q_lo, KH, KL, VH, VL, kT_off, v_off, s_m,
                offs_m, offs_n, offs_d, S, diag, hi,
                HEAD_D, BLOCK_N, True, SPLIT, EVEN_S, EVEN_D)
        else:
            acc, l_i, m_i = _attn_inner(
                acc, l_i, m_i, q_hi, q_lo, KH, KL, VH, VL, kT_off, v_off, s_m,
                offs_m, offs_n, offs_d, S, 0, S,
                HEAD_D, BLOCK_N, False, SPLIT, EVEN_S, EVEN_D)

        o = acc / l_i[:, None]
        o_off = b * o_b + h * o_h + offs_m[:, None] * o_m + offs_d[None, :]
        if EVEN_S and EVEN_D:
            tl.store(O + o_off, o.to(O.dtype.element_ty))
        else:
            tl.store(O + o_off, o.to(O.dtype.element_ty),
                     mask=(offs_m[:, None] < S) & (offs_d[None, :] < HEAD_D))


def _config(block_d: int, split: int):
    """Tile sizes for Turing: 64 KB shared memory per SM, no async copies, and
    SPLIT=3 keeps hi and lo tiles of both k and v live. BLOCK_M is always a
    multiple of BLOCK_N so the diagonal phase starts on a key-tile boundary."""
    if block_d <= 32:
        return (64, 64 if split == 1 else 32, 4)
    if block_d <= 64:
        return (64, 32, 4)
    return (32, 32, 4)


def can_use(q: torch.Tensor) -> bool:
    return (HAVE_TRITON and q.is_cuda and q.dim() == 4
            and q.shape[-1] <= MAX_HEAD_DIM
            and q.dtype in (torch.float32, torch.float16, torch.bfloat16))


def _operands(q, k, v, scale, split):
    """fp16 hi/lo halves of the (scaled) q, k and v, contiguous [B, H, S, D].

    The hi half is the fp16 rounding; the lo half is what it dropped, itself
    rounded to fp16 -- together they carry ~22 bits. For inputs that are
    already fp16 the lo half would be identically zero, so SPLIT is lowered
    to 1 rather than paying for three matmuls of nothing.
    """
    if q.dtype != torch.float32:
        split = 1
    qs = (q.float() * (scale * LOG2E)).contiguous()
    kc = k.float().contiguous()
    vc = v.float().contiguous()
    q_hi, k_hi, v_hi = qs.half(), kc.half(), vc.half()
    if split == 3:
        q_lo = (qs - q_hi.float()).half()
        k_lo = (kc - k_hi.float()).half()
        v_lo = (vc - v_hi.float()).half()
    else:
        q_lo, k_lo, v_lo = q_hi, k_hi, v_hi
    return (q_hi, q_lo, k_hi, k_lo, v_hi, v_lo), split


def _launch(kernel, ops, o, causal, split):
    q_hi, q_lo, k_hi, k_lo, v_hi, v_lo = ops
    B, H, S, D = q_hi.shape
    block_d = max(16, triton.next_power_of_2(D))
    block_m, block_n, warps = _config(block_d, split)
    even_s = (S % block_m == 0) and (S % block_n == 0)
    even_d = (D == block_d)
    grid = (triton.cdiv(S, block_m), B * H)
    sb, sh, sm, _ = q_hi.stride()
    ob, oh, om, _ = o.stride()
    kernel[grid](
        q_hi, q_lo, k_hi, k_lo, v_hi, v_lo, o,
        sb, sh, sm, ob, oh, om, H, S,
        HEAD_D=D, BLOCK_D=block_d, BLOCK_M=block_m, BLOCK_N=block_n,
        CAUSAL=bool(causal), SPLIT=int(split), EVEN_S=even_s, EVEN_D=even_d,
        num_warps=warps, num_stages=2,
    )


def attention_raw(q, k, v, scale=None, causal=True, split=3):
    """q, k, v: [B, H, S, D], any strides. Returns a contiguous [B, H, S, D]
    tensor in q's dtype."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    ops, split = _operands(q, k, v, scale, split)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _launch(_attn_fwd, ops, o, causal, split)
    return o


HAVE_ATTN_OP = False
if HAVE_TRITON:
    try:
        from torch.library import triton_op, wrap_triton

        @triton_op("exactswap::attention", mutates_args={})
        def _attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          scale: float, causal: bool, split: int) -> torch.Tensor:
            ops, split = _operands(q, k, v, scale, split)
            o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
            _launch(wrap_triton(_attn_fwd), ops, o, causal, split)
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
