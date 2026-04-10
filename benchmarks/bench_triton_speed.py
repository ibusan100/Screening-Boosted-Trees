"""
M6 Speed Benchmark: NumPy (CPU) vs Triton (GPU) for histogram + screening.

Two sections:
  1. Kernel-level microbenchmark  — raw throughput of build_histogram + screening_split
  2. E2E booster benchmark        — ScreeningBooster(device="cpu") vs device="cuda"
                                    on California Housing

Run:
    python benchmarks/bench_triton_speed.py
"""

import time
from pathlib import Path
import json

import numpy as np

# ---- check CUDA availability ----
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

if not HAS_CUDA:
    print("CUDA not available — Triton benchmarks skipped.")
    print("NumPy baselines will still run.")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ibu_boost import ScreeningBooster, ScreeningParams
from ibu_boost.screening_split import (
    build_histogram_numpy, build_missing_stats, screening_split_numpy
)

PARAMS = ScreeningParams(s_w=-2.0, s_r=-6.0, lam=1.0)
SEEDS  = [0, 1, 2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timeit(fn, warmup=2, repeats=10):
    for _ in range(warmup):
        fn()
    if HAS_CUDA:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if HAS_CUDA:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


# ---------------------------------------------------------------------------
# 1. Kernel microbenchmark
# ---------------------------------------------------------------------------

def micro_benchmark():
    print("\n" + "="*60)
    print("Kernel microbenchmark  (N=65536, F=8, B=255)")
    print("="*60)

    rng  = np.random.default_rng(0)
    N, F, B = 65536, 8, 255
    X    = rng.integers(0, B, size=(N, F), dtype=np.int32)
    # ~10% missing
    miss = rng.random((N, F)) < 0.10
    X[miss] = -1
    g    = rng.standard_normal(N).astype(np.float32)
    h    = rng.uniform(0.5, 1.5, N).astype(np.float32)
    nid  = np.zeros(N, dtype=np.int32)

    # NumPy baseline
    def numpy_path():
        hG, hH = build_histogram_numpy(X, g, h, nid, 1, B)
        Gm, Hm = build_missing_stats(X, g, h, nid, 1)
        screening_split_numpy(hG, hH, PARAMS, Gm, Hm)

    t_np = timeit(numpy_path, warmup=3, repeats=20)
    print(f"  NumPy  : {t_np*1000:.2f} ms / call")

    if HAS_CUDA:
        from ibu_boost.kernels.screening_split_triton import (
            build_histogram_triton, build_missing_stats_triton, screening_split_triton
        )
        X_t   = torch.from_numpy(X).cuda()
        g_t   = torch.from_numpy(g).cuda()
        h_t   = torch.from_numpy(h).cuda()
        nid_t = torch.from_numpy(nid).cuda()

        def triton_path():
            hG, hH = build_histogram_triton(X_t, g_t, h_t, nid_t, 1, B)
            Gm, Hm = build_missing_stats_triton(X_t, g_t, h_t, nid_t, 1)
            screening_split_triton(hG, hH, PARAMS, Gm, Hm)
            torch.cuda.synchronize()

        t_tr = timeit(triton_path, warmup=5, repeats=50)
        speedup = t_np / t_tr
        print(f"  Triton : {t_tr*1000:.2f} ms / call")
        print(f"  Speedup: {speedup:.1f}x")
        return {"numpy_ms": t_np*1000, "triton_ms": t_tr*1000, "speedup": speedup}
    else:
        return {"numpy_ms": t_np*1000}


# ---------------------------------------------------------------------------
# 2. E2E booster benchmark — California Housing
# ---------------------------------------------------------------------------

def booster_benchmark():
    print("\n" + "="*60)
    print("E2E ScreeningBooster - California Housing")
    print("  n_estimators=100, max_depth=6, oblivious")
    print("="*60)

    X_full, y_full = fetch_california_housing(return_X_y=True)
    y_full = y_full.astype(np.float32)

    times_cpu, times_gpu, rmse_cpu, rmse_gpu = [], [], [], []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_full, test_size=0.2, random_state=seed
        )

        # CPU
        m_cpu = ScreeningBooster(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            params=PARAMS, tree_type="oblivious", objective="regression",
            device="cpu",
        )
        t0 = time.perf_counter()
        m_cpu.fit(X_tr, y_tr)
        times_cpu.append(time.perf_counter() - t0)
        preds = m_cpu.predict(X_te)
        rmse_cpu.append(float(np.sqrt(np.mean((y_te - preds)**2))))
        print(f"  [seed={seed}] CPU  RMSE={rmse_cpu[-1]:.4f}  time={times_cpu[-1]:.2f}s")

        if HAS_CUDA:
            m_gpu = ScreeningBooster(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                params=PARAMS, tree_type="oblivious", objective="regression",
                device="cuda",
            )
            if HAS_CUDA:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            m_gpu.fit(X_tr, y_tr)
            if HAS_CUDA:
                torch.cuda.synchronize()
            times_gpu.append(time.perf_counter() - t0)
            preds_gpu = m_gpu.predict(X_te)
            rmse_gpu.append(float(np.sqrt(np.mean((y_te - preds_gpu)**2))))
            print(f"  [seed={seed}] CUDA RMSE={rmse_gpu[-1]:.4f}  time={times_gpu[-1]:.2f}s")

    print(f"\n=== Summary ===")
    print(f"  CPU  RMSE={np.mean(rmse_cpu):.4f}+-{np.std(rmse_cpu):.4f}  "
          f"time={np.mean(times_cpu):.2f}+-{np.std(times_cpu):.2f}s")
    if HAS_CUDA:
        speedup = np.mean(times_cpu) / np.mean(times_gpu)
        print(f"  CUDA RMSE={np.mean(rmse_gpu):.4f}+-{np.std(rmse_gpu):.4f}  "
              f"time={np.mean(times_gpu):.2f}+-{np.std(times_gpu):.2f}s")
        print(f"  E2E speedup: {speedup:.2f}x")
        return {
            "cpu":  {"rmse_mean": float(np.mean(rmse_cpu)), "rmse_std": float(np.std(rmse_cpu)),
                     "time_mean": float(np.mean(times_cpu))},
            "cuda": {"rmse_mean": float(np.mean(rmse_gpu)), "rmse_std": float(np.std(rmse_gpu)),
                     "time_mean": float(np.mean(times_gpu))},
            "e2e_speedup": float(speedup),
        }
    else:
        return {
            "cpu": {"rmse_mean": float(np.mean(rmse_cpu)), "rmse_std": float(np.std(rmse_cpu)),
                    "time_mean": float(np.mean(times_cpu))},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if HAS_CUDA:
        dev = torch.cuda.get_device_name(0)
        print(f"GPU: {dev}")
        print(f"PyTorch: {torch.__version__}")
    else:
        print("Running CPU-only (no CUDA)")

    micro = micro_benchmark()
    e2e   = booster_benchmark()

    out = Path(__file__).parent / "results_triton_speed.json"
    out.write_text(json.dumps({"micro": micro, "e2e": e2e}, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
