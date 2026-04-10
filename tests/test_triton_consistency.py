"""
Triton vs NumPy numerical consistency tests — CLAUDE.md Inv-2.

Layer          | atol   | rtol   | Notes
---------------|--------|--------|----------------------------------------------
Kernel A       | 1e-3   | 1e-3   | atomic_add is non-deterministic on GPU
Kernel A2      | 1e-3   | 1e-3   | same (missing scatter, atomic_add)
Kernel B       | 1e-6   | 1e-6   | deterministic (same histogram input)
End-to-end     | 1e-3   | 1e-3   | Kernel A atol dominates

All tests are skipped when CUDA is not available.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

try:
    from ibu_boost.kernels.screening_split_triton import (
        build_histogram_triton,
        build_missing_stats_triton,
        screening_split_triton,
    )
except ImportError as e:
    pytest.skip(f"Triton not available: {e}", allow_module_level=True)

from ibu_boost.screening_split import (
    ScreeningParams,
    build_histogram_numpy,
    build_missing_stats,
    screening_split_numpy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda"


def make_data(N=4096, F=4, B=32, missing_frac=0.0, seed=0):
    """Generate binned features, gradients, hessians, node IDs.

    missing_frac: fraction of (sample, feat) entries set to MISSING_BIN=-1.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, B, size=(N, F), dtype=np.int32)
    g = rng.standard_normal(N).astype(np.float32)
    h = rng.uniform(0.5, 1.5, size=N).astype(np.float32)
    nid = np.zeros(N, dtype=np.int32)
    if missing_frac > 0.0:
        miss_mask = rng.random((N, F)) < missing_frac
        X[miss_mask] = -1
    return X, g, h, nid


def to_cuda(*arrs):
    return [torch.from_numpy(a).to(DEVICE) for a in arrs]


def shared_histogram(N=4096, F=4, B=32, missing_frac=0.0, seed=0):
    """Build one histogram with NumPy, return both np and cuda versions."""
    X, g, h, nid = make_data(N=N, F=F, B=B, missing_frac=missing_frac, seed=seed)
    hG_np, hH_np = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=B)
    hG_t = torch.from_numpy(hG_np).to(DEVICE)
    hH_t = torch.from_numpy(hH_np).to(DEVICE)
    Gm_np, Hm_np = build_missing_stats(X, g, h, nid, num_nodes=1)
    Gm_t = torch.from_numpy(Gm_np).to(DEVICE)
    Hm_t = torch.from_numpy(Hm_np).to(DEVICE)
    return hG_np, hH_np, hG_t, hH_t, Gm_np, Hm_np, Gm_t, Hm_t


# ---------------------------------------------------------------------------
# Kernel A: build_histogram_triton  (atol=1e-3)
# ---------------------------------------------------------------------------


def test_kernel_a_shape():
    X, g, h, nid = make_data(N=2048, F=4, B=32)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, hH_t = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=32)
    assert hG_t.shape == (1, 4, 32)
    assert hH_t.shape == (1, 4, 32)


def test_kernel_a_vs_numpy_g():
    X, g, h, nid = make_data(N=4096, F=4, B=32)
    hG_np, _ = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, _ = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=32)
    np.testing.assert_allclose(hG_t.cpu().numpy(), hG_np, atol=1e-3, rtol=1e-3)


def test_kernel_a_vs_numpy_h():
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


def test_kernel_a_skips_missing_bin():
    """MISSING_BIN=-1 samples must not appear in the histogram."""
    N, F, B = 2048, 4, 32
    X, g, h, nid = make_data(N=N, F=F, B=B, missing_frac=0.2)
    hG_np, hH_np = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=B)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, hH_t = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=B)
    np.testing.assert_allclose(hG_t.cpu().numpy(), hG_np, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(hH_t.cpu().numpy(), hH_np, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Kernel A2: build_missing_stats_triton  (atol=1e-3)
# ---------------------------------------------------------------------------


def test_kernel_a2_shape():
    X, g, h, nid = make_data(N=2048, F=4, B=32, missing_frac=0.2)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    Gm_t, Hm_t = build_missing_stats_triton(X_t, g_t, h_t, nid_t, num_nodes=1)
    assert Gm_t.shape == (1, 4)
    assert Hm_t.shape == (1, 4)


def test_kernel_a2_vs_numpy():
    N, F, B = 4096, 4, 32
    X, g, h, nid = make_data(N=N, F=F, B=B, missing_frac=0.15)
    Gm_np, Hm_np = build_missing_stats(X, g, h, nid, num_nodes=1)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    Gm_t, Hm_t = build_missing_stats_triton(X_t, g_t, h_t, nid_t, num_nodes=1)
    np.testing.assert_allclose(Gm_t.cpu().numpy(), Gm_np, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(Hm_t.cpu().numpy(), Hm_np, atol=1e-3, rtol=1e-3)


def test_kernel_a2_no_missing_gives_zeros():
    X, g, h, nid = make_data(N=2048, F=4, B=32, missing_frac=0.0)
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    Gm_t, Hm_t = build_missing_stats_triton(X_t, g_t, h_t, nid_t, num_nodes=1)
    assert Gm_t.abs().max().item() == 0.0
    assert Hm_t.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Kernel B: screening_split_triton  (atol=1e-6 — same histogram fed to both)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s_w,s_r", [(-2.0, 0.0), (-4.0, -2.0), (-1.0, 0.5)])
def test_kernel_b_rho_matches_numpy(s_w, s_r):
    hG_np, hH_np, hG_t, hH_t, _, _, _, _ = shared_histogram(F=4, B=32)
    params = ScreeningParams(s_w=s_w, s_r=s_r, lam=1.0)

    out_np = screening_split_numpy(hG_np, hH_np, params)
    out_tr = screening_split_triton(hG_t, hH_t, params)

    np.testing.assert_allclose(
        out_tr["best_rho"].cpu().numpy(), out_np["best_rho"],
        atol=1e-6, rtol=1e-6,
        err_msg=f"best_rho mismatch s_w={s_w}, s_r={s_r}",
    )


@pytest.mark.parametrize("s_w,s_r", [(-2.0, 0.0), (-4.0, -2.0)])
def test_kernel_b_best_feat_bin_matches_numpy(s_w, s_r):
    hG_np, hH_np, hG_t, hH_t, _, _, _, _ = shared_histogram(F=4, B=32)
    params = ScreeningParams(s_w=s_w, s_r=s_r, lam=1.0)

    out_np = screening_split_numpy(hG_np, hH_np, params)
    out_tr = screening_split_triton(hG_t, hH_t, params)

    np.testing.assert_array_equal(out_tr["best_feat"].cpu().numpy(), out_np["best_feat"])
    np.testing.assert_array_equal(out_tr["best_bin"].cpu().numpy(),  out_np["best_bin"])


def test_kernel_b_accepted_mask_matches():
    hG_np, hH_np, hG_t, hH_t, _, _, _, _ = shared_histogram()
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)
    out_np = screening_split_numpy(hG_np, hH_np, params)
    out_tr = screening_split_triton(hG_t, hH_t, params)
    np.testing.assert_array_equal(out_tr["accepted_mask"].cpu().numpy(), out_np["accepted_mask"])


@pytest.mark.parametrize("s_w,s_r", [(-2.0, 0.0), (-4.0, -2.0)])
def test_kernel_b_with_missing_rho_matches(s_w, s_r):
    """Kernel B with G_miss/H_miss must match NumPy reference (atol=1e-6)."""
    hG_np, hH_np, hG_t, hH_t, Gm_np, Hm_np, Gm_t, Hm_t = shared_histogram(
        F=4, B=32, missing_frac=0.15
    )
    params = ScreeningParams(s_w=s_w, s_r=s_r, lam=1.0)

    out_np = screening_split_numpy(hG_np, hH_np, params, Gm_np, Hm_np)
    out_tr = screening_split_triton(hG_t, hH_t, params, Gm_t, Hm_t)

    np.testing.assert_allclose(
        out_tr["best_rho"].cpu().numpy(), out_np["best_rho"],
        atol=1e-6, rtol=1e-6,
        err_msg=f"best_rho (with missing) mismatch s_w={s_w}, s_r={s_r}",
    )
    np.testing.assert_array_equal(
        out_tr["best_feat"].cpu().numpy(), out_np["best_feat"],
    )
    np.testing.assert_array_equal(
        out_tr["best_bin"].cpu().numpy(), out_np["best_bin"],
    )


def test_kernel_b_best_default_dir_matches():
    """best_default_dir must agree with NumPy best_default_dir (atol=0)."""
    hG_np, hH_np, hG_t, hH_t, Gm_np, Hm_np, Gm_t, Hm_t = shared_histogram(
        F=4, B=32, missing_frac=0.2
    )
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)

    out_np = screening_split_numpy(hG_np, hH_np, params, Gm_np, Hm_np)
    out_tr = screening_split_triton(hG_t, hH_t, params, Gm_t, Hm_t)

    np.testing.assert_array_equal(
        out_tr["best_default_dir"].cpu().numpy().astype(np.int8),
        out_np["best_default_dir"],
    )


# ---------------------------------------------------------------------------
# End-to-end: Kernel A + A2 → B  (atol=1e-3)
# ---------------------------------------------------------------------------


def test_e2e_best_rho_matches():
    """Full A+A2→B pipeline: build hist+missing with Triton, split with Triton."""
    X, g, h, nid = make_data(N=4096, F=4, B=32, missing_frac=0.15)
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)

    # NumPy path
    hG_np, hH_np = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    Gm_np, Hm_np = build_missing_stats(X, g, h, nid, num_nodes=1)
    out_np = screening_split_numpy(hG_np, hH_np, params, Gm_np, Hm_np)

    # Triton path
    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, hH_t = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=32)
    Gm_t, Hm_t = build_missing_stats_triton(X_t, g_t, h_t, nid_t, num_nodes=1)
    out_tr = screening_split_triton(hG_t, hH_t, params, Gm_t, Hm_t)

    np.testing.assert_allclose(
        out_tr["best_rho"].cpu().numpy(), out_np["best_rho"],
        atol=1e-3, rtol=1e-3,
        err_msg="E2E best_rho mismatch exceeds atol=1e-3",
    )


def test_e2e_no_missing():
    """E2E without missing values: histogram-only path."""
    X, g, h, nid = make_data(N=4096, F=4, B=32, missing_frac=0.0)
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)

    hG_np, hH_np = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    out_np = screening_split_numpy(hG_np, hH_np, params)

    X_t, g_t, h_t, nid_t = to_cuda(X, g, h, nid)
    hG_t, hH_t = build_histogram_triton(X_t, g_t, h_t, nid_t, num_nodes=1, num_bins=32)
    out_tr = screening_split_triton(hG_t, hH_t, params)

    np.testing.assert_allclose(
        out_tr["best_rho"].cpu().numpy(), out_np["best_rho"],
        atol=1e-3, rtol=1e-3,
    )
    np.testing.assert_array_equal(out_tr["best_feat"].cpu().numpy(), out_np["best_feat"])
    np.testing.assert_array_equal(out_tr["best_bin"].cpu().numpy(),  out_np["best_bin"])
