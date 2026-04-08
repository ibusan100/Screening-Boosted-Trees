"""
Triton vs NumPy numerical consistency tests — CLAUDE.md Inv-2.

Layer          | atol   | rtol   | Notes
---------------|--------|--------|----------------------------------------------
Kernel A       | 1e-3   | 1e-3   | atomic_add is non-deterministic on GPU
Kernel B       | 1e-6   | 1e-6   | deterministic (same histogram input)
End-to-end     | 1e-3   | 1e-3   | Kernel A atol dominates

All tests are skipped when CUDA is not available.
"""

import numpy as np
import pytest

# Attempt to import torch and triton; skip entire module gracefully if absent.
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

try:
    from sbt.kernels.screening_split_triton import (
        build_histogram_triton,
        screening_split_triton,
    )
except ImportError as e:
    pytest.skip(f"Triton not available: {e}", allow_module_level=True)

from sbt.screening_split import (
    ScreeningParams,
    build_histogram_numpy,
    screening_split_numpy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = "cuda"
RNG = np.random.default_rng(123)


def make_data(N=4096, F=4, B=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, B, size=(N, F), dtype=np.int32)
    g = rng.standard_normal(N).astype(np.float32)
    h = rng.uniform(0.5, 1.5, size=N).astype(np.float32)
    nid = np.zeros(N, dtype=np.int32)
    return X, g, h, nid


def to_cuda(*arrs):
    return [torch.from_numpy(a).to(DEVICE) for a in arrs]


# ---------------------------------------------------------------------------
# Kernel A: build_histogram  (atol=1e-3)
# ---------------------------------------------------------------------------


def test_kernel_a_shape():
    X, g, h, nid = make_data(N=2048, F=4, B=32)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, hH_t = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=32)
    assert hG_t.shape == (1, 4, 32)
    assert hH_t.shape == (1, 4, 32)


def test_kernel_a_vs_numpy_g(benchmark_atol=1e-3):
    """Per-feature gradient sums must match (within atomic-add tolerance)."""
    X, g, h, nid = make_data(N=4096, F=4, B=32)
    # NumPy reference
    hG_np, _ = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    # Triton
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, _ = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=32)
    hG_cpu = hG_t.cpu().numpy()
    np.testing.assert_allclose(hG_cpu, hG_np, atol=benchmark_atol, rtol=benchmark_atol)


def test_kernel_a_vs_numpy_h():
    """Hessian histogram matches (h_i are positive, so atol is tighter in practice)."""
    X, g, h, nid = make_data(N=4096, F=4, B=32)
    _, hH_np = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    _, hH_t = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=32)
    np.testing.assert_allclose(hH_t.cpu().numpy(), hH_np, atol=1e-3, rtol=1e-3)


def test_kernel_a_two_nodes():
    N, F, B = 2048, 4, 32
    rng = np.random.default_rng(7)
    X = rng.integers(0, B, size=(N, F), dtype=np.int32)
    g = rng.standard_normal(N).astype(np.float32)
    h = np.ones(N, dtype=np.float32)
    nid = (np.arange(N) % 2).astype(np.int32)

    hG_np, hH_np = build_histogram_numpy(X, g, h, nid, num_nodes=2, num_bins=B)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, hH_t = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=2, num_bins=B)

    np.testing.assert_allclose(hG_t.cpu().numpy(), hG_np, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(hH_t.cpu().numpy(), hH_np, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Kernel B: screening_split  (atol=1e-6  — same histogram fed to both)
# ---------------------------------------------------------------------------


def _shared_histogram(N=4096, F=4, B=32, seed=0):
    """Build one histogram with NumPy, return both np and cuda versions."""
    X, g, h, nid = make_data(N=N, F=F, B=B, seed=seed)
    hG_np, hH_np = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=B)
    hG_t = torch.from_numpy(hG_np).to(DEVICE)
    hH_t = torch.from_numpy(hH_np).to(DEVICE)
    return hG_np, hH_np, hG_t, hH_t


@pytest.mark.parametrize("s_w,s_r", [(-2.0, 0.0), (-4.0, -2.0), (-1.0, 0.5)])
def test_kernel_b_rho_matches_numpy(s_w, s_r):
    """Kernel B on exact same histogram must match NumPy to atol=1e-6."""
    hG_np, hH_np, hG_t, hH_t = _shared_histogram(F=4, B=32)
    params = ScreeningParams(s_w=s_w, s_r=s_r, lam=1.0)

    out_np = screening_split_numpy(hG_np, hH_np, params)
    out_tr = screening_split_triton(hG_t, hH_t, params)

    np.testing.assert_allclose(
        out_tr["best_rho"].cpu().numpy(),
        out_np["best_rho"],
        atol=1e-6, rtol=1e-6,
        err_msg=f"best_rho mismatch at s_w={s_w}, s_r={s_r}",
    )


@pytest.mark.parametrize("s_w,s_r", [(-2.0, 0.0), (-4.0, -2.0)])
def test_kernel_b_best_feat_matches_numpy(s_w, s_r):
    hG_np, hH_np, hG_t, hH_t = _shared_histogram(F=4, B=32)
    params = ScreeningParams(s_w=s_w, s_r=s_r, lam=1.0)

    out_np = screening_split_numpy(hG_np, hH_np, params)
    out_tr = screening_split_triton(hG_t, hH_t, params)

    np.testing.assert_array_equal(
        out_tr["best_feat"].cpu().numpy(),
        out_np["best_feat"],
    )
    np.testing.assert_array_equal(
        out_tr["best_bin"].cpu().numpy(),
        out_np["best_bin"],
    )


def test_kernel_b_accepted_mask_matches():
    hG_np, hH_np, hG_t, hH_t = _shared_histogram()
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)

    out_np = screening_split_numpy(hG_np, hH_np, params)
    out_tr = screening_split_triton(hG_t, hH_t, params)

    np.testing.assert_array_equal(
        out_tr["accepted_mask"].cpu().numpy(),
        out_np["accepted_mask"],
    )


# ---------------------------------------------------------------------------
# End-to-end: Kernel A → B  (atol=1e-3)
# ---------------------------------------------------------------------------


def test_e2e_best_rho_matches():
    """Full A→B pipeline: build hist with Triton, split with Triton, compare to NumPy."""
    X, g, h, nid = make_data(N=4096, F=4, B=32)
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)

    # NumPy path
    hG_np, hH_np = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    out_np = screening_split_numpy(hG_np, hH_np, params)

    # Triton path (Kernel A → B)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, hH_t = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=32)
    out_tr = screening_split_triton(hG_t, hH_t, params)

    np.testing.assert_allclose(
        out_tr["best_rho"].cpu().numpy(),
        out_np["best_rho"],
        atol=1e-3, rtol=1e-3,
        err_msg="E2E best_rho mismatch exceeds atol=1e-3",
    )
