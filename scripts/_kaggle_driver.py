# ===== driver (appended to the self-contained kernel) =====
# References names defined earlier in the combined module: TransformerConfig,
# BaselineTransformer, UserOptimizedTransformer, copy_model_weights,
# generate_random_case, compare_outputs.

_RESULTS = []  # (idx, pass, max_abs, max_rel, base_ms, opt_ms, speedup, note)
_ABL_RESULTS = []  # (idx, stage, pass, max_abs, max_rel, base_ms, opt_ms, speedup)


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



# ---- stage ablation ---------------------------------------------------------
# Each stage is selected purely by environment variable, with no code edits, so
# the numbers below and the delivered path are literally the same code.
_ABL_CONFIGS = [
    ("sdpa",              {"T3_AUTOCAST": "off",  "T3_COMPILE": "0"}),
    ("sdpa+compile",      {"T3_AUTOCAST": "off",  "T3_COMPILE": "1"}),
    ("sdpa+fp16",         {"T3_AUTOCAST": "fp16", "T3_COMPILE": "0"}),
    ("sdpa+compile+fp16", {"T3_AUTOCAST": "fp16", "T3_COMPILE": "1"}),
]
_ABL_SHAPES = [
    (1, 64, 128, 4, 128, 4, 128),
    (12, 64, 128, 4, 32, 4, 128),
    (8, 64, 1024, 4, 128, 4, 1024),
    (13, 64, 128, 4, 1024, 4, 128),
]


def _ablation(device):
    import os
    import torch
    dtype = torch.float32
    print("stage ablation: shape,stage,pass,max_abs,max_rel,baseline_ms,opt_ms,speedup",
          flush=True)
    for (idx, b, d, h, sq, l, f) in _ABL_SHAPES:
        cfg = TransformerConfig(batch_size=b, seq_len=sq, d_model=d, num_heads=h,
                                ffn_dim=f, num_layers=l, causal=True)
        cfg.validate()
        baseline = BaselineTransformer(cfg).to(device, dtype).eval()
        xt, mt = generate_random_case(cfg, device, dtype, 101234, 0.0, 1.0)
        bms = _bench(baseline, xt, mt)
        for name, env in _ABL_CONFIGS:
            for k, v in env.items():
                os.environ[k] = v
            try:
                # Construct AFTER setting the env: __init__ and the one-shot
                # _plan() are what read these knobs.
                opt = UserOptimizedTransformer(cfg)
                copy_model_weights(baseline, opt, strict=True)
                opt = opt.to(device, dtype).eval()
                ok, mabs, mrel = _accuracy(baseline, opt, cfg, device, dtype)
                oms = _bench(opt, xt, mt)
                print(f"ABL,{idx},{name},{'PASS' if ok else 'FAIL'},{mabs:.3g},"
                      f"{mrel:.3g},{bms:.4f},{oms:.4f},{bms/oms:.3f}", flush=True)
                _ABL_RESULTS.append((idx, name, "PASS" if ok else "FAIL",
                                     f"{mabs:.3g}", f"{mrel:.3g}", f"{bms:.4f}",
                                     f"{oms:.4f}", f"{bms/oms:.3f}"))
            except Exception as e:
                print(f"ABL,{idx},{name},ERROR,,,,,{str(e)[:80]}", flush=True)
                _ABL_RESULTS.append((idx, name, "ERROR", "", "", "", "", str(e)[:60]))
            finally:
                try:
                    del opt
                except Exception:
                    pass
                torch.cuda.empty_cache()
        del baseline, xt, mt
        torch.cuda.empty_cache()
    # Restore the shipped defaults for anything that runs after us.
    os.environ["T3_AUTOCAST"] = "off"
    os.environ["T3_COMPILE"] = "1"


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

    if only in ("all", "ablation"):
        print("\n##### STAGE ABLATION #####", flush=True)
        try:
            _ablation(device)
        except Exception as e:
            print("ABLATION ERROR:", str(e)[:200], flush=True)

    print("\n##### SHAPE 14 : optimized-only (baseline infeasible ~20.5 TB) #####", flush=True)
    try:
        if only in ("1-13", "ablation"):
            raise _Skip
        _shape14(device)
    except _Skip:
        print("skipped for this T3_ONLY selector", flush=True)
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
    if _ABL_RESULTS:
        print("# --- stage ablation ---", flush=True)
        print("abl_shape,stage,pass,max_abs,max_rel,baseline_ms,opt_ms,speedup",
              flush=True)
        for r in _ABL_RESULTS:
            print(",".join(str(x) for x in r), flush=True)
    print("=================== END SUMMARY ===================", flush=True)


_main()
