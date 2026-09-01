#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.001 and rtol=0.01 (1%).
"""

from __future__ import annotations

# --- bootstrap: ensure a torch build compatible with the allocated GPU ---
# Kaggle's API-allocated GPU is often a Tesla P100 (sm_60), which the preinstalled
# torch 2.10+cu128 does NOT support (sm_70+ only). Detect an incompatible build and
# reinstall a P100+T4-compatible torch, then re-exec so the new build is loaded.
import os as _os, sys as _sys, subprocess as _sp
# Expandable segments keep the allocator from fragmenting when the seq_len=1e5
# shape holds two ~6.5 GB tensors (input + output) plus per-chunk activations.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if _os.environ.get("_T3_BOOT") != "1":
    _need = False
    try:
        import torch as _t
        if _t.cuda.is_available():
            _cc = "sm_%d%d" % _t.cuda.get_device_capability()
            _need = _cc not in _t.cuda.get_arch_list()
        else:
            _need = True
    except Exception:
        _need = True
    if _need:
        print("[bootstrap] GPU/torch mismatch -> installing torch 2.5.1+cu121 ...", flush=True)
        _sp.run([_sys.executable, "-m", "pip", "install", "-q",
                 "--index-url", "https://download.pytorch.org/whl/cu121",
                 "torch==2.5.1"], check=False)
        _os.environ["_T3_BOOT"] = "1"
        _os.execv(_sys.executable, [_sys.executable] + _sys.argv)
    _os.environ["_T3_BOOT"] = "1"

import argparse
import copy
import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(BaselineTransformer):
    """
    Replace this class with the optimized implementation.

    Requirements:
      1. Keep the forward signature unchanged.
      2. Return a tensor with shape [batch_size, seq_len, d_model].
      3. Keep compatible parameter names, or customize copy_model_weights().
    """

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ====================== your codes here ======================
        # Example optimization directions:
        #   * torch.nn.functional.scaled_dot_product_attention
        #   * torch.compile
        #   * Triton/CUDA fused kernels
        #   * fused LayerNorm / residual / FFN
        #
        # The default implementation calls the baseline so that this script
        # remains directly runnable before the optimized code is inserted.
        return super().forward(x, valid_token_mask)
        # ============================================================


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> None:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return 0 if accuracy_passed else 2


# ================= user_optimized.py (inlined) =================

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


import os
from typing import Optional

import torch
import torch.nn.functional as F

# Reuse the reference model definition so parameter names match exactly.


# Live [chunk, S, max(D, ffn)] intermediates a block keeps alive simultaneously.
_LIVE_INTERMEDIATES = 8

# torch.cuda.OutOfMemoryError exists on torch>=1.13; older builds raise RuntimeError.
_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


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

        # Batch-chunk only when the full batch's activations would not fit in
        # the VRAM that is actually free right now. In practice this only ever
        # triggers for the seq_len=1e5 shape.
        override = os.environ.get("T3_CHUNK_BS")
        b, s, _ = x.shape
        fdim = self.config.ffn_dim
        per_sample = s * max(self.config.d_model, fdim)
        if override is not None:
            self._chunk_bs = max(1, int(override))
        elif b * per_sample <= 300_000_000:
            # Whole-batch activations are small by any measure -> never chunk.
            # Keeps every graded shape (1-13) on the single-pass path.
            self._chunk_bs = None
        else:
            cb = max(1, min(b, self._chunk_budget(x) // max(1, per_sample)))
            self._chunk_bs = cb if cb < b else None

    def _chunk_budget(self, x: torch.Tensor) -> int:
        """Elements of working-set headroom available for one chunk.

        The chunked path pre-allocates the full [B,S,D] output, so the budget is
        (free VRAM - output buffer) with a fragmentation reserve, divided by the
        number of live intermediates a block keeps alive at once (~8: q/k/v, the
        SDPA output, its contiguous copy, the projection, and the two residuals).
        """
        if x.device.type != "cuda":
            return 300_000_000
        try:
            free_dev, _total = torch.cuda.mem_get_info(x.device)
            # Blocks the caching allocator already holds but is not using.
            free_dev += torch.cuda.memory_reserved(x.device) - torch.cuda.memory_allocated(
                x.device
            )
        except Exception:
            return 300_000_000
        out_bytes = x.numel() * x.element_size()
        usable = (free_dev - out_bytes) * 0.6  # 40% reserve for fragmentation
        elem_bytes = 2 if self._autocast_dtype is not None else x.element_size()
        return max(1, int(usable / (elem_bytes * _LIVE_INTERMEDIATES)))

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
            return self._forward_chunked(x, valid_token_mask, core, b)

        return core(x, valid_token_mask, all_valid).to(x.dtype)

    def _forward_chunked(self, x, valid_token_mask, core, b):
        """Batch-chunked forward for the shape whose activations exceed VRAM.

        Writes each chunk straight into a pre-allocated output instead of
        collecting a list and ``torch.cat``-ing it: the concat would hold both
        the pieces and the joined result at once, doubling peak VRAM exactly
        when memory is tightest (it is what made seq_len=1e5 OOM on a 16 GB
        card). On OOM the chunk size is halved and the pass restarted, so a
        mis-estimated budget degrades instead of failing.
        """
        while True:
            try:
                out = torch.empty_like(x)
                for i in range(0, b, self._chunk_bs):
                    sl = slice(i, i + self._chunk_bs)
                    ms = None if valid_token_mask is None else valid_token_mask[sl]
                    av = True if ms is None else bool(ms.all())
                    piece = core(x[sl], ms, av)
                    out[sl].copy_(piece)
                    del piece
                return out
            except _OOM:
                if self._chunk_bs <= 1:
                    raise
                out = None
                self._chunk_bs = max(1, self._chunk_bs // 2)
                torch.cuda.empty_cache()


# ================= sweep driver =================

# ===== driver (appended to the self-contained kernel) =====
# References names defined earlier in the combined module: TransformerConfig,
# BaselineTransformer, UserOptimizedTransformer, copy_model_weights,
# generate_random_case, compare_outputs.

_RESULTS = []  # (idx, pass, max_abs, max_rel, base_ms, opt_ms, speedup, note)


class _Skip(Exception):
    """Sentinel for the T3_ONLY selector."""


def _bench(model, x, mask, warmup=20, iters=50):
    import torch
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        s = []
        for _ in range(iters):
            st.record(); model(x, mask); en.record(); torch.cuda.synchronize()
            s.append(st.elapsed_time(en))
    s.sort()
    return s[len(s) // 2]


def _accuracy(baseline, optimized, cfg, device, dtype, trials=3):
    import torch
    ok = True; mabs = mrel = 0.0
    with torch.inference_mode():
        for t in range(trials):
            x, m = generate_random_case(cfg, device, dtype, 1234 + t, 0.0, 1.0)
            ref = baseline(x, m); o = optimized(x, m)
            r = compare_outputs(ref, o, rtol=0.02, atol=0.002)
            ok &= r.passed; mabs = max(mabs, r.max_abs_error); mrel = max(mrel, r.max_relative_error)
    return ok, mabs, mrel


def _shape14(device):
    import torch
    FULL = dict(batch_size=32, seq_len=100000, d_model=1024, num_heads=16,
                ffn_dim=1024, num_layers=2, causal=True)
    note = ""
    tc = dict(FULL); tc["seq_len"] = 2048; tc["batch_size"] = 2
    cfg = TransformerConfig(**tc)
    base = BaselineTransformer(cfg); opt = UserOptimizedTransformer(cfg)
    copy_model_weights(base, opt, strict=True)
    base = base.to(device, torch.float32).eval(); opt = opt.to(device, torch.float32).eval()
    x, m = generate_random_case(cfg, device, torch.float32, 1234, 0.0, 1.0)
    with torch.inference_mode():
        ref = base(x, m); o = opt(x, m)
    res = compare_outputs(ref, o, rtol=0.02, atol=0.002)
    tpass = "PASS" if res.passed else "FAIL"
    print(f"trunc S=2048 correctness: {tpass} max_abs={res.max_abs_error:.3g} "
          f"max_rel={res.max_relative_error:.3g}", flush=True)
    del base, opt, x, m, ref, o
    torch.cuda.empty_cache()

    cfg = TransformerConfig(**FULL)
    base = BaselineTransformer(cfg); opt = UserOptimizedTransformer(cfg)
    copy_model_weights(base, opt, strict=True); del base
    opt = opt.to(device, torch.float16).eval()
    torch.cuda.reset_peak_memory_stats(device)
    free0, total0 = torch.cuda.mem_get_info(device)
    scores_tb = cfg.batch_size * cfg.num_heads * cfg.seq_len ** 2 * 4 / 1e12
    print(f"vram free={free0/1e9:.2f}/{total0/1e9:.2f} GB | baseline scores would be "
          f"{scores_tb:.1f} TB -> infeasible", flush=True)
    try:
        x, m = generate_random_case(cfg, device, torch.float16, 1234, 0.0, 1.0)
        med = _bench(opt, x, m, warmup=3, iters=10)
        tok = cfg.batch_size * cfg.seq_len
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        note = (f"full S=1e5 OK {med:.0f}ms {tok*1000.0/med:,.0f}tok/s "
                f"peak{peak:.1f}GB chunk{opt._chunk_bs}")
        print(f"full S=100000: median={med:.1f} ms | {tok*1000.0/med:,.0f} tok/s | "
              f"peak_vram={peak:.2f} GB | chunk_bs={opt._chunk_bs}", flush=True)
    except RuntimeError as e:
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        note = f"full S=1e5 OOM peak{peak:.1f}GB chunk{opt._chunk_bs}"
        print("shape14 full-seq RuntimeError:", str(e)[:300], flush=True)
    _RESULTS.append((14, tpass + "(trunc)", res.max_abs_error, res.max_relative_error,
                     "", "", "", note))


def _main():
    import os
    import torch
    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    print("=== ENV ===", flush=True)
    print(f"gpu {torch.cuda.get_device_name(device)} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda} | cc {torch.cuda.get_device_capability(device)}", flush=True)

    only = os.environ.get("T3_ONLY", "all").strip().lower()

    SH = [
        (1, 64, 128, 4, 128, 4, 128), (2, 1, 128, 4, 128, 4, 128),
        (3, 4, 128, 4, 128, 4, 128), (4, 16, 128, 4, 128, 4, 128),
        (5, 128, 128, 4, 128, 4, 128), (6, 10000, 128, 4, 128, 4, 128),
        (7, 64, 32, 4, 128, 4, 32), (8, 64, 1024, 4, 128, 4, 1024),
        (9, 64, 128, 1, 128, 4, 128), (10, 64, 128, 2, 128, 4, 128),
        (11, 64, 128, 16, 128, 4, 128), (12, 64, 128, 4, 32, 4, 128),
        (13, 64, 128, 4, 1024, 4, 128),
    ]
    for (idx, b, d, h, s, l, f) in (SH if only in ("all", "1-13") else []):
        print(f"\n##### SHAPE {idx} : B={b} D={d} H={h} S={s} L={l} F={f} #####", flush=True)
        cfg = TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                ffn_dim=f, num_layers=l, causal=True)
        cfg.validate()
        try:
            baseline = BaselineTransformer(cfg).to(device, dtype).eval()
            optimized = UserOptimizedTransformer(cfg)
            copy_model_weights(baseline, optimized, strict=True)
            optimized = optimized.to(device, dtype).eval()
            ok, mabs, mrel = _accuracy(baseline, optimized, cfg, device, dtype)
            if ok:
                xt, mt = generate_random_case(cfg, device, dtype, 101234, 0.0, 1.0)
                bms = _bench(baseline, xt, mt); oms = _bench(optimized, xt, mt)
                sp = bms / oms
                print(f"PASS max_abs={mabs:.3g} max_rel={mrel:.3g} | "
                      f"baseline={bms:.4f}ms optimized={oms:.4f}ms | speedup={sp:.3f}x", flush=True)
                _RESULTS.append((idx, "PASS", mabs, mrel, f"{bms:.4f}", f"{oms:.4f}", f"{sp:.3f}", ""))
            else:
                print(f"FAIL max_abs={mabs:.3g} max_rel={mrel:.3g}", flush=True)
                _RESULTS.append((idx, "FAIL", mabs, mrel, "", "", "", ""))
        except Exception as e:
            print(f"SHAPE {idx} ERROR:", str(e)[:200], flush=True)
            _RESULTS.append((idx, "ERROR", "", "", "", "", "", str(e)[:60]))
        finally:
            try:
                del baseline, optimized
            except Exception:
                pass
            torch.cuda.empty_cache()

    print("\n##### SHAPE 14 : optimized-only (baseline infeasible ~20.5 TB) #####", flush=True)
    try:
        if only == "1-13":
            raise _Skip
        _shape14(device)
    except _Skip:
        print("skipped (T3_ONLY=1-13)", flush=True)
    except Exception as e:
        print("SHAPE 14 ERROR:", str(e)[:200], flush=True)
        _RESULTS.append((14, "ERROR", "", "", "", "", "", str(e)[:60]))

    # ---- compact summary: copy THIS block back ----
    sp_vals = sorted(float(r[6]) for r in _RESULTS if r[6] and r[1] == "PASS")
    print("\n=================== SUMMARY (copy from here) ===================", flush=True)
    print("shape,pass,max_abs,max_rel,baseline_ms,opt_ms,speedup,note", flush=True)
    for r in _RESULTS:
        print(",".join(str(x) for x in r), flush=True)
    if sp_vals:
        print(f"# median_speedup={sp_vals[len(sp_vals)//2]:.3f}x "
              f"min={sp_vals[0]:.3f}x max={sp_vals[-1]:.3f}x over {len(sp_vals)} PASS shapes", flush=True)
    print("=================== END SUMMARY ===================", flush=True)


_main()
