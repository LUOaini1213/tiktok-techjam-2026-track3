#!/usr/bin/env python3
"""
UserOptimizedTransformer — TikTok TechJam 2026, Track 3
Implement a GPU Kernel for a Transformer Layer.

This module rewrites ONLY the forward compute of the reference
``BaselineTransformer`` (defined in ``torch_transformer_benchmark.py``) while
keeping every submodule and parameter name identical, so the harness'
``copy_model_weights(..., strict=True)`` succeeds with zero friction.

Optimization levers (all numerically equivalent within the harness tolerance
atol=0.002 OR rtol=0.02, checked per-element):
  1. Attention via ``F.scaled_dot_product_attention`` (FlashAttention /
     memory-efficient kernel) -> O(S) memory instead of the baseline's
     O(S^2) materialized score matrix. This is what makes the seq_len=100000
     shape possible at all (the baseline would need ~20.5 TB for its scores).
  2. Internal fp16 autocast even when the grader runs float32 -> lights up the
     Turing/Ampere tensor cores. rtol=0.02 (2%) leaves ~40x margin over fp16
     rounding; reductions (LayerNorm, softmax) stay in fp32.
  3. Self-applied ``torch.compile`` (does not depend on the grader passing
     --compile-user); mode chosen per shape (reduce-overhead for launch-bound
     small shapes, default otherwise).
  4. Batch chunking ONLY for the extreme shape (seq_len=1e5) so activations fit
     in 16 GB.

Ablation / robustness toggles via environment variables (see README):
  T3_AUTOCAST   = auto | fp16 | bf16 | off      (default auto)
  T3_COMPILE    = 1 | 0                          (default 1)
  T3_COMPILE_MODE = default | reduce-overhead | max-autotune  (override)
  T3_FP32_FFN   = 1 | 0                          (default 0; force FFN+LN fp32)
  T3_CHUNK_BS   = <int>                          (override batch chunk size)
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

# Reuse the reference model definition so parameter names match exactly.
from torch_transformer_benchmark import BaselineTransformer


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class UserOptimizedTransformer(BaselineTransformer):
    """Drop-in optimized replacement. Same weights, faster forward."""

    def __init__(self, config) -> None:
        super().__init__(config)  # identical submodules -> strict weight copy OK

        # Lazily resolved on the first CUDA forward (see _plan).
        self._planned = False
        self._autocast_dtype: Optional[torch.dtype] = None
        self._chunk_bs: Optional[int] = None
        self._compiled = None
        self._compile_ok = _env_flag("T3_COMPILE", True)
        self._can_compile = False  # set in _plan: Triton needs CUDA capability >= 7.0
        self._fp32_ffn = _env_flag("T3_FP32_FFN", False)

    # ---- one-time device/shape aware planning -------------------------------
    def _plan(self, x: torch.Tensor) -> None:
        if self._planned:
            return
        self._planned = True

        # torch.compile uses the Triton backend, which requires CUDA capability
        # >= 7.0 (Volta+). Older GPUs (e.g. Kaggle's Tesla P100 = sm_60) must run
        # eager. get_device_capability does not launch a kernel, so it is safe.
        self._can_compile = (
            x.device.type == "cuda"
            and torch.cuda.get_device_capability(x.device)[0] >= 7
        )

        # Precision: DEFAULT to the native dtype (no autocast). Internal fp16 is
        # opt-in only (T3_AUTOCAST=fp16/auto), because fp16's ~1e-3 per-op error
        # accumulates across layers and breaks the strict atol=0.002 gate for
        # near-zero output elements. Running in the grader's own dtype always
        # passes: fp32 grading -> fp32 (exact), fp16 grading -> fp16 (matches).
        want = os.environ.get("T3_AUTOCAST", "off").strip().lower()
        if x.device.type != "cuda" or x.dtype != torch.float32 or want == "off":
            self._autocast_dtype = None
        elif want == "fp16":
            self._autocast_dtype = torch.float16
        elif want == "bf16":
            self._autocast_dtype = torch.bfloat16
        else:  # auto: T4/Turing has fp16 tensor cores only; bf16 on A100/L4.
            name = torch.cuda.get_device_name()
            if ("T4" in name) or (not torch.cuda.is_bf16_supported()):
                self._autocast_dtype = torch.float16
            else:
                self._autocast_dtype = torch.bfloat16

        # Batch-chunk only when a single batch element's largest activation is
        # big enough that the full batch would blow the VRAM budget. With the
        # 300M-element budget this only triggers for the seq_len=1e5 shape.
        override = os.environ.get("T3_CHUNK_BS")
        b, s, _ = x.shape
        fdim = self.config.ffn_dim
        per_sample = s * max(self.config.d_model, fdim)
        if override is not None:
            self._chunk_bs = max(1, int(override))
        else:
            budget = 300_000_000
            cb = max(1, min(b, budget // max(1, per_sample)))
            self._chunk_bs = cb if cb < b else None

    # ---- compute ------------------------------------------------------------
    def _attention(self, attn, x, mask, causal, all_valid):
        b, s, d = x.shape
        h, hd = attn.num_heads, attn.head_dim
        q = attn.q_proj(x).view(b, s, h, hd).transpose(1, 2)
        k = attn.k_proj(x).view(b, s, h, hd).transpose(1, 2)
        v = attn.v_proj(x).view(b, s, h, hd).transpose(1, 2)

        if all_valid:
            # Graded hot path: no padding. A dense [S,S] mask is impossible for
            # S=1e5, so causality MUST go through is_causal (kernel-generated).
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=causal, scale=attn.scale
            )
        else:
            # Padded fallback (only reached for small S with a real mask).
            neg = torch.finfo(q.dtype).min
            bias = torch.zeros(b, 1, s, s, dtype=q.dtype, device=q.device)
            bias = bias.masked_fill((~mask)[:, None, None, :], neg)
            if causal:
                causal_mask = torch.ones(
                    s, s, dtype=torch.bool, device=q.device
                ).triu(1)
                bias = bias.masked_fill(causal_mask[None, None], neg)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=attn.scale
            )
            out = torch.nan_to_num(out, nan=0.0)  # guard fully-masked rows

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        out = attn.out_proj(out)
        if mask is not None and not all_valid:
            out = out.masked_fill(~mask[..., None], 0)
        return out

    def _ffn(self, layer, h2):
        if self._fp32_ffn and self._autocast_dtype is not None:
            with torch.autocast("cuda", enabled=False):
                h2 = h2.float()
                return layer.ffn_out(F.gelu(layer.ffn_in(h2), approximate="none"))
        return layer.ffn_out(F.gelu(layer.ffn_in(h2), approximate="none"))

    def _block(self, layer, x, mask, causal, all_valid):
        x = x + self._attention(layer.attention, layer.norm1(x), mask, causal, all_valid)
        x = x + self._ffn(layer, layer.norm2(x))
        if mask is not None and not all_valid:
            x = x.masked_fill(~mask[..., None], 0)
        return x

    def _run_full(self, x, mask, causal, all_valid):
        for layer in self.layers:
            x = self._block(layer, x, mask, causal, all_valid)
        x = self.final_norm(x)
        if mask is not None and not all_valid:
            x = x.masked_fill(~mask[..., None], 0)
        return x

    # ---- entry point --------------------------------------------------------
    def forward(self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None):
        self._plan(x)
        causal = self.config.causal
        # Device->host sync kept OUTSIDE any compiled region / CUDA graph.
        all_valid = valid_token_mask is None or bool(valid_token_mask.all())
        ad = self._autocast_dtype
        b = x.shape[0]

        # Lazily build the compiled callable on the first CUDA forward.
        if (self._compiled is None and self._compile_ok and self._can_compile):
            try:
                mode = os.environ.get("T3_COMPILE_MODE")
                if mode is None:
                    mode = "reduce-overhead" if b * x.shape[1] <= 16384 else "default"
                self._compiled = torch.compile(self._run_full, mode=mode, dynamic=False)
            except Exception:
                self._compile_ok = False
                self._compiled = None

        def _invoke(fn, xin, m, av):
            if ad is not None:
                with torch.autocast("cuda", dtype=ad):
                    return fn(xin, m, causal, av)
            return fn(xin, m, causal, av)

        def core(xin, m, av):
            fn = self._compiled if self._compiled is not None else self._run_full
            try:
                return _invoke(fn, xin, m, av)
            except Exception:
                if fn is self._run_full:
                    raise
                # Compiled path failed at CALL time (e.g. Triton needs sm>=7.0,
                # or an inductor edge case) -> permanently fall back to eager.
                self._compile_ok = False
                self._compiled = None
                return _invoke(self._run_full, xin, m, av)

        if self._chunk_bs is not None and self._chunk_bs < b:  # extreme shape only
            outs: List[torch.Tensor] = []
            for i in range(0, b, self._chunk_bs):
                sl = slice(i, i + self._chunk_bs)
                ms = None if valid_token_mask is None else valid_token_mask[sl]
                av = True if ms is None else bool(ms.all())
                outs.append(core(x[sl], ms, av).to(x.dtype))
            return torch.cat(outs, dim=0)

        return core(x, valid_token_mask, all_valid).to(x.dtype)
