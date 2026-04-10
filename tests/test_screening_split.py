"""
Tests for the NumPy screening split reference implementation.

Layer structure (CLAUDE.md Inv-2):
  - Kernel B (post-histogram screening transform): atol=1e-6
  - build_histogram: tested for shape / sum correctness (not diff vs Triton yet)
"""

import numpy as np
import pytest

from ibu_boost.screening_split import (
    ScreeningParams,
    build_histogram_numpy,
    screening_split_numpy,
)


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_data(N=2048, F=4, B=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, B, size=(N, F), dtype=np.int32)
    g = rng.standard_normal(N).astype(np.float32)
    h = rng.uniform(0.5, 1.5, size=N).astype(np.float32)
    nid = np.zeros(N, dtype=np.int32)
    return X, g, h, nid


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------


def test_histogram_shape():
    X, g, h, nid = make_data(N=1000, F=4, B=16)
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=16)
    assert hG.shape == (1, 4, 16)
    assert hH.shape == (1, 4, 16)


def test_histogram_sums_match_inputs():
    # Each sample contributes to F histogram entries (one per feature),
    # so hG.sum() == F * g.sum() and hH.sum() == F * h.sum().
    X, g, h, nid = make_data(N=1000, F=4, B=16)
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=16)
    np.testing.assert_allclose(hG.sum(), 4 * g.sum(), rtol=1e-5)
    np.testing.assert_allclose(hH.sum(), 4 * h.sum(), rtol=1e-5)
    # Per-feature slice must match exactly
    np.testing.assert_allclose(hG[0, 0, :].sum(), g.sum(), rtol=1e-5)


def test_histogram_two_nodes():
    rng = np.random.default_rng(7)
    N, F, B = 400, 3, 8
    X = rng.integers(0, B, size=(N, F), dtype=np.int32)
    g = rng.standard_normal(N).astype(np.float32)
    h = np.ones(N, dtype=np.float32)
    nid = (np.arange(N) % 2).astype(np.int32)
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=2, num_bins=B)
    # Each node gets N/2 samples × F features in the histogram sums.
    # hH[node].sum() == F * n_samples_in_node == 3 * 200 == 600
    assert abs(hH[0].sum() - F * 200) < 10
    assert abs(hH[1].sum() - F * 200) < 10
    # Per-feature check: each feature slice of node 0 should sum to ~200
    assert abs(hH[0, 0, :].sum() - 200) < 10


# ---------------------------------------------------------------------------
# Screening split (Kernel B equivalent) — atol 1e-6 (CLAUDE.md Inv-2)
# ---------------------------------------------------------------------------


def test_screening_split_deterministic():
    """Same histogram → identical result across two calls (Kernel B is deterministic)."""
    X, g, h, nid = make_data()
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    params = ScreeningParams(s_w=0.0, s_r=0.0, lam=1.0)
    r1 = screening_split_numpy(hG, hH, params)
    r2 = screening_split_numpy(hG, hH, params)
    np.testing.assert_array_equal(r1["best_feat"], r2["best_feat"])
    np.testing.assert_array_equal(r1["best_bin"],  r2["best_bin"])
    np.testing.assert_allclose(r1["best_rho"], r2["best_rho"], atol=1e-6)


def test_rho_bounded():
    """rho values must lie in [0, 1]."""
    X, g, h, nid = make_data()
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    for s_r in [-2.0, 0.0, 2.0]:
        params = ScreeningParams(s_w=0.0, s_r=s_r, lam=1.0)
        out = screening_split_numpy(hG, hH, params)
        assert out["rho"].min() >= 0.0 - 1e-7
        assert out["rho"].max() <= 1.0 + 1e-7


def test_last_bin_always_zero():
    """Last bin has no right child; rho must be exactly 0 there."""
    X, g, h, nid = make_data(B=32)
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    params = ScreeningParams(s_w=-4.0, s_r=-4.0, lam=1e-3)  # very permissive
    out = screening_split_numpy(hG, hH, params)
    np.testing.assert_array_equal(out["rho"][:, :, -1], 0.0)


def test_high_sr_rejects_all():
    """Very large s_r (large r) → acceptance width 1/r ≈ 0 → all rejected."""
    X, g, h, nid = make_data()
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    params = ScreeningParams(s_w=0.0, s_r=10.0, lam=1.0)
    out = screening_split_numpy(hG, hH, params)
    assert not out["accepted_mask"][0], "Expected all splits to be rejected with huge s_r"


def test_low_sr_accepts_some():
    """Very small s_r (r ≈ 1, acceptance width ≈ 1) → most or all accepted."""
    X, g, h, nid = make_data()
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=32)
    params = ScreeningParams(s_w=-6.0, s_r=-10.0, lam=1e-3)
    out = screening_split_numpy(hG, hH, params)
    assert out["accepted_mask"][0], "Expected at least one split to be accepted"
    assert out["best_rho"][0] > 0.0


def test_trim_square_formula_spot_check():
    """Manually verify the Trim-and-Square formula on a known histogram.

    After H_total normalisation:
      H_total = 4  →  norm_gain = raw_gain / 4
      bin 0: raw=9.333  →  norm=2.333  →  s=1-exp(-2.333/tau)
      bin 1: raw=4.0    →  norm=1.0
      bin 2: raw=5.333  →  norm=1.333
      bin 0 has highest norm_gain → best_bin must be 0.
    """
    hG = np.array([[[3.0, -1.0, 2.0, 0.0]]], dtype=np.float32)  # [1, 1, 4]
    hH = np.array([[[1.0,  1.0, 1.0, 1.0]]], dtype=np.float32)
    # Use tau=1.0 (s_w=0), r=2 (s_r=0), lam=0
    params = ScreeningParams(s_w=0.0, s_r=0.0, lam=0.0, eps=1e-9)
    # norm_gain[0] = 9.333/4 = 2.333; s[0] = 1 - exp(-2.333) ≈ 0.903
    # rho[0] = max(1 - 2*(1-0.903), 0)^2 = max(0.806, 0)^2 ≈ 0.65 > 0
    out = screening_split_numpy(hG, hH, params)
    assert out["best_bin"][0] == 0, f"Expected best_bin=0, got {out['best_bin'][0]}"
    assert out["best_rho"][0] > 0.0, f"Expected rho > 0 at bin 0, got {out['best_rho'][0]}"
    assert out["rho"][0, 0, -1] == 0.0  # last bin always 0


# ---------------------------------------------------------------------------
# accept_rate monotonicity w.r.t. s_r
# ---------------------------------------------------------------------------


def test_accept_rate_decreases_with_sr():
    """Higher s_r (stricter r) should monotonically reduce or maintain accept_rate."""
    X, g, h, nid = make_data(N=4096, F=8, B=64)
    hG, hH = build_histogram_numpy(X, g, h, nid, num_nodes=1, num_bins=64)
    prev_rate = 1.1
    for s_r in [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
        params = ScreeningParams(s_w=-1.0, s_r=s_r, lam=1.0)
        out = screening_split_numpy(hG, hH, params)
        rate = (out["rho"] > 0).mean()
        assert rate <= prev_rate + 1e-6, (
            f"accept_rate should not increase as s_r goes from prev to {s_r}: "
            f"{prev_rate:.4f} → {rate:.4f}"
        )
        prev_rate = rate
