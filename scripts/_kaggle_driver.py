# ===== driver (appended to the self-contained kernel) =====
# References names defined earlier in the combined module: TransformerConfig,
# BaselineTransformer, UserOptimizedTransformer, copy_model_weights,
# generate_random_case, compare_outputs, run_accuracy_tests, benchmark_models.

def _shape14(device):
    import torch
    FULL = dict(batch_size=32, seq_len=100000, d_model=1024, num_heads=16,
                ffn_dim=1024, num_layers=2, causal=True)

    # (B) correctness by construction at a truncated seq_len the baseline can run.
    tc = dict(FULL); tc["seq_len"] = 2048
    cfg = TransformerConfig(**tc)
    base = BaselineTransformer(cfg); opt = UserOptimizedTransformer(cfg)
    copy_model_weights(base, opt, strict=True)
    base = base.to(device, torch.float32).eval(); opt = opt.to(device, torch.float32).eval()
    x, m = generate_random_case(cfg, device, torch.float32, 1234, 0.0, 1.0)
    with torch.inference_mode():
        ref = base(x, m); o = opt(x, m)
    res = compare_outputs(ref, o, rtol=0.02, atol=0.002)
    print(f"trunc S=2048 correctness: {'PASS' if res.passed else 'FAIL'} "
          f"max_abs={res.max_abs_error:.3g} max_rel={res.max_relative_error:.3g}", flush=True)
    del base, opt, x, m, ref, o
    torch.cuda.empty_cache()

    # (A) full-seq timing, optimized only, fp16, batch-chunked.
    cfg = TransformerConfig(**FULL)
    base = BaselineTransformer(cfg); opt = UserOptimizedTransformer(cfg)
    copy_model_weights(base, opt, strict=True); del base
    opt = opt.to(device, torch.float16).eval()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        x, m = generate_random_case(cfg, device, torch.float16, 1234, 0.0, 1.0)
        with torch.inference_mode():
            for _ in range(5):
                opt(x, m)
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(20):
                st.record(); opt(x, m); en.record(); torch.cuda.synchronize()
                samples.append(st.elapsed_time(en))
        samples.sort(); med = samples[len(samples) // 2]
        tok = cfg.batch_size * cfg.seq_len
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"full S=100000: median={med:.1f} ms | {tok*1000.0/med:,.0f} tok/s | "
              f"peak_vram={peak:.2f} GB | chunk_bs={opt._chunk_bs}", flush=True)
    except RuntimeError as e:
        print("shape14 full-seq RuntimeError:", str(e)[:300], flush=True)


def _main():
    import torch
    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    print("=== ENV ===", flush=True)
    print(f"gpu {torch.cuda.get_device_name(device)} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda} | bf16 {torch.cuda.is_bf16_supported()}", flush=True)

    SH = [
        (1, 64, 128, 4, 128, 4, 128), (2, 1, 128, 4, 128, 4, 128),
        (3, 4, 128, 4, 128, 4, 128), (4, 16, 128, 4, 128, 4, 128),
        (5, 128, 128, 4, 128, 4, 128), (6, 10000, 128, 4, 128, 4, 128),
        (7, 64, 32, 4, 128, 4, 32), (8, 64, 1024, 4, 128, 4, 1024),
        (9, 64, 128, 1, 128, 4, 128), (10, 64, 128, 2, 128, 4, 128),
        (11, 64, 128, 16, 128, 4, 128), (12, 64, 128, 4, 32, 4, 128),
        (13, 64, 128, 4, 1024, 4, 128),
    ]
    for (idx, b, d, h, s, l, f) in SH:
        print(f"\n##### SHAPE {idx} : B={b} D={d} H={h} S={s} L={l} F={f} #####", flush=True)
        cfg = TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                ffn_dim=f, num_layers=l, causal=True)
        cfg.validate()
        try:
            baseline = BaselineTransformer(cfg)
            optimized = UserOptimizedTransformer(cfg)
            copy_model_weights(baseline, optimized, strict=True)
            baseline = baseline.to(device=device, dtype=dtype).eval()
            optimized = optimized.to(device=device, dtype=dtype).eval()
            passed = run_accuracy_tests(baseline, optimized, cfg, device, dtype,
                                        3, 1234, 0.0, 1.0, 0.02, 0.002)
            if passed:
                benchmark_models(baseline, optimized, cfg, device, dtype,
                                 1234, 0.0, 1.0, 20, 100, 3)
            else:
                print("accuracy FAILED -> timing skipped", flush=True)
        except RuntimeError as e:
            print(f"SHAPE {idx} RuntimeError:", str(e)[:300], flush=True)
        finally:
            for _n in ("baseline", "optimized"):
                if _n in dir():
                    pass
            try:
                del baseline, optimized
            except Exception:
                pass
            torch.cuda.empty_cache()

    print("\n##### SHAPE 14 : optimized-only (baseline infeasible ~20.5 TB scores) #####",
          flush=True)
    _shape14(device)
    print("\n=== ALL DONE ===", flush=True)


_main()
