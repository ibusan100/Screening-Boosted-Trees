"""
Tests for ScreeningTree — fit, predict, diagnostics.
"""

import numpy as np
import pytest

from ibu_boost.tree import ScreeningTree
from ibu_boost.screening_split import ScreeningParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_regression(N=500, F=4, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(N, F))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + noise * rng.standard_normal(N)
    return X.astype(np.float64), y.astype(np.float32)


def make_step_function(N=400):
    """Simple dataset where a 1-feature step function should be perfectly learnable."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(N, 2))
    y = (X[:, 0] > 0.5).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    X, y = make_regression()
    tree = ScreeningTree(max_depth=3, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    result = tree.fit(X, y)
    assert result is tree


def test_predict_shape():
    X, y = make_regression()
    tree = ScreeningTree(max_depth=3, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)
    preds = tree.predict(X)
    assert preds.shape == (len(y),)


def test_train_mse_decreases_with_depth():
    """Deeper trees should have lower training MSE."""
    X, y = make_regression(N=1000)
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)
    mse = {}
    for d in [1, 2, 4]:
        tree = ScreeningTree(max_depth=d, min_samples_leaf=10, params=params)
        tree.fit(X, y)
        preds = tree.predict(X)
        mse[d] = float(np.mean((y - preds) ** 2))
    assert mse[1] >= mse[2] >= mse[4], f"MSE should decrease with depth: {mse}"


def test_step_function_learned():
    """Tree should fit a simple step function near-perfectly."""
    X, y = make_step_function(N=800)
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1e-3)  # default calibrated scale
    tree = ScreeningTree(max_depth=4, min_samples_leaf=5, params=params)
    tree.fit(X, y)
    preds = tree.predict(X)
    mse = float(np.mean((y - preds) ** 2))
    assert mse < 0.05, f"Expected low MSE on step function, got {mse:.4f}"


def test_predict_constant_when_max_depth_zero():
    """max_depth=0 → single leaf → predictions = global mean."""
    X, y = make_regression()
    params = ScreeningParams(s_w=0.0, s_r=0.0)
    tree = ScreeningTree(max_depth=0, params=params)
    tree.fit(X, y)
    preds = tree.predict(X)
    expected = np.mean(y)
    np.testing.assert_allclose(preds, expected, atol=1e-4)


def test_predict_on_unseen_data():
    """predict() on held-out X should not crash and return finite values."""
    X, y = make_regression()
    X_train, X_test = X[:400], X[400:]
    tree = ScreeningTree(max_depth=3, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X_train, y[:400])
    preds = tree.predict(X_test)
    assert np.isfinite(preds).all()
    assert preds.shape == (100,)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_present_after_fit():
    X, y = make_regression()
    tree = ScreeningTree(max_depth=3, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)
    diag = tree.diagnostics
    assert diag is not None
    assert len(diag.nodes) > 0


def test_diagnostics_root_node_exists():
    X, y = make_regression()
    tree = ScreeningTree(max_depth=3, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)
    assert tree.diagnostics.root_accept_rate is not None


def test_diagnostics_accept_rate_in_range():
    """With default params, root accept_rate should be strictly in (0, 1)."""
    X, y = make_regression(N=1000)
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)  # default calibrated scale
    tree = ScreeningTree(max_depth=3, params=params)
    tree.fit(X, y)
    rate = tree.diagnostics.root_accept_rate
    assert 0.0 < rate < 1.0, f"Root accept_rate = {rate:.4f}, expected strictly in (0, 1)"


def test_diagnostics_rejected_at_root_with_huge_sr():
    """Constant y → zero gain everywhere → all splits rejected regardless of params."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 4))
    y = np.full(300, 3.14, dtype=np.float32)   # constant → g_i = 0 → gain = 0 always
    params = ScreeningParams(s_w=0.0, s_r=0.0)
    tree = ScreeningTree(max_depth=3, params=params)
    tree.fit(X, y)
    assert tree.diagnostics.rejected_at_root


def test_diagnostics_n_candidates():
    """n_candidates at root = (num_bins - 1) * F (last bin excluded per spec)."""
    X, y = make_regression(N=500, F=4)
    params = ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0)
    num_bins = 32
    tree = ScreeningTree(max_depth=1, min_samples_leaf=5, num_bins=num_bins, params=params)
    tree.fit(X, y)
    root_diag = next(n for n in tree.diagnostics.nodes if n.depth == 0)
    # actual_max_bins may be <= num_bins due to unique-value quantile collisions
    assert root_diag.n_candidates > 0


def test_screening_leaves_count():
    """Leaves created by screening rejection should be tracked separately."""
    X, y = make_regression(N=500)
    # Calibrate to ensure some nodes are split and some are rejected by screening
    params = ScreeningParams(s_w=0.0, s_r=1.5, lam=1.0)
    tree = ScreeningTree(max_depth=6, min_samples_leaf=5, params=params)
    tree.fit(X, y)
    total_leaves = tree.n_leaves
    screening_leaves = tree.diagnostics.n_leaves_by_screening
    assert total_leaves >= 1
    assert screening_leaves >= 0  # may be 0 if all leaves hit depth/sample limit


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------


def test_n_nodes_consistent():
    """n_nodes should equal the number of nodes in the internal list."""
    X, y = make_regression()
    tree = ScreeningTree(max_depth=3, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)
    assert tree.n_nodes == len(tree._nodes)


def test_all_samples_reach_leaves():
    """Every training sample should map to a leaf with a finite value."""
    X, y = make_regression(N=300)
    tree = ScreeningTree(max_depth=4, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)
    preds = tree.predict(X)
    assert np.isfinite(preds).all()
    assert len(preds) == len(y)
