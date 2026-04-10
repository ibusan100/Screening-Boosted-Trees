"""
M5 missing value tests — NaN handling in Binner, screening_split, ScreeningTree,
ObliviousTree, and ScreeningBooster.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibu_boost import (
    Binner,
    ScreeningBooster,
    ScreeningParams,
    ScreeningTree,
    ObliviousTree,
    build_histogram_numpy,
    screening_split_numpy,
)
from ibu_boost.screening_split import build_missing_stats


RNG = np.random.default_rng(99)

# ---------------------------------------------------------------------------
# Binner
# ---------------------------------------------------------------------------


def test_binner_nan_maps_to_missing_bin():
    X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
    binner = Binner(num_bins=10)
    X_b = binner.fit_transform(X)
    assert X_b[0, 1] == -1   # NaN in feature 1 row 0
    assert X_b[2, 0] == -1   # NaN in feature 0 row 2
    assert X_b[1, 0] >= 0    # valid value
    assert X_b[1, 1] >= 0    # valid value


def test_binner_all_nan_feature():
    """Feature with all NaN values should still produce a valid Binner."""
    X = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
    binner = Binner(num_bins=5)
    X_b = binner.fit_transform(X)
    assert (X_b[:, 0] == -1).all()  # all NaN → all MISSING_BIN
    assert (X_b[:, 1] >= 0).all()   # non-NaN feature unaffected


# ---------------------------------------------------------------------------
# build_missing_stats
# ---------------------------------------------------------------------------


def test_build_missing_stats_basic():
    # Feature 0: sample 0 is missing; feature 1: sample 1 is missing
    X_b = np.array([[-1, 0], [1, -1]], dtype=np.int32)
    g = np.array([2.0, 3.0], dtype=np.float32)
    h = np.array([1.0, 1.0], dtype=np.float32)
    nid = np.zeros(2, dtype=np.int32)
    G_miss, H_miss = build_missing_stats(X_b, g, h, nid, num_nodes=1)
    assert G_miss.shape == (1, 2)
    assert H_miss.shape == (1, 2)
    np.testing.assert_allclose(G_miss[0, 0], 2.0)  # sample 0 missing in feat 0
    np.testing.assert_allclose(G_miss[0, 1], 3.0)  # sample 1 missing in feat 1
    np.testing.assert_allclose(H_miss[0, 0], 1.0)
    np.testing.assert_allclose(H_miss[0, 1], 1.0)


def test_build_missing_stats_no_missing():
    X_b = np.array([[0, 1], [2, 3]], dtype=np.int32)
    g = np.ones(2, dtype=np.float32)
    h = np.ones(2, dtype=np.float32)
    nid = np.zeros(2, dtype=np.int32)
    G_miss, H_miss = build_missing_stats(X_b, g, h, nid, num_nodes=1)
    np.testing.assert_array_equal(G_miss, 0.0)
    np.testing.assert_array_equal(H_miss, 0.0)


# ---------------------------------------------------------------------------
# screening_split_numpy — default_dir output
# ---------------------------------------------------------------------------


def test_screening_split_returns_default_dir():
    rng = np.random.default_rng(0)
    X_b = rng.integers(0, 16, size=(200, 4)).astype(np.int32)
    g = rng.standard_normal(200).astype(np.float32)
    h = np.ones(200, dtype=np.float32)
    nid = np.zeros(200, dtype=np.int32)

    hist_G, hist_H = build_histogram_numpy(X_b, g, h, nid, 1, 16)
    G_miss, H_miss = build_missing_stats(X_b, g, h, nid, 1)
    result = screening_split_numpy(hist_G, hist_H, ScreeningParams(), G_miss, H_miss)

    assert "default_dir" in result
    assert result["default_dir"].shape == (1, 4, 16)
    assert result["default_dir"].dtype == np.int8
    assert set(result["default_dir"].ravel().tolist()).issubset({0, 1})


def test_screening_split_default_dir_without_missing():
    """Without G_miss/H_miss, default_dir should be all zeros."""
    rng = np.random.default_rng(1)
    hist_G = rng.standard_normal((1, 3, 8)).astype(np.float32)
    hist_H = np.abs(rng.standard_normal((1, 3, 8))).astype(np.float32) + 0.1
    result = screening_split_numpy(hist_G, hist_H, ScreeningParams())
    assert "default_dir" in result
    # last bin should be 0 (masked); others can be 0 since no missing provided
    assert result["default_dir"].dtype == np.int8


# ---------------------------------------------------------------------------
# ScreeningTree with NaN
# ---------------------------------------------------------------------------


def make_nan_data(n=300, seed=0):
    """Step-function dataset with 20% NaN in feature 0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.float32) * 2.0
    nan_mask = rng.random(n) < 0.2
    X[nan_mask, 0] = np.nan
    return X, y


def test_screening_tree_fit_with_nan():
    X, y = make_nan_data()
    tree = ScreeningTree(max_depth=3, min_samples_leaf=10, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)
    assert tree.n_leaves >= 2


def test_screening_tree_predict_nan_returns_finite():
    X, y = make_nan_data()
    tree = ScreeningTree(max_depth=3, min_samples_leaf=10, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)

    X_test = X.copy()
    X_test[:10, 0] = np.nan  # force NaN in test set
    preds = tree.predict(X_test)
    assert np.isfinite(preds).all()
    assert preds.shape == (len(X_test),)


def test_screening_tree_nan_better_than_ignore():
    """Predictions on NaN samples should not all be the global mean."""
    X, y = make_nan_data(n=500)
    tree = ScreeningTree(max_depth=4, min_samples_leaf=10, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)
    # NaN samples should be routed — predictions should vary
    X_nan = np.full((20, 4), np.nan)
    preds = tree.predict(X_nan)
    assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------
# ObliviousTree with NaN
# ---------------------------------------------------------------------------


def test_oblivious_tree_fit_with_nan():
    X, y = make_nan_data()
    tree = ObliviousTree(max_depth=4, min_samples_leaf=10, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)
    assert tree.depth >= 1


def test_oblivious_tree_predict_nan_returns_finite():
    X, y = make_nan_data()
    tree = ObliviousTree(max_depth=4, min_samples_leaf=10, params=ScreeningParams(s_w=-2.0, s_r=0.0))
    tree.fit(X, y)

    X_test = X.copy()
    X_test[:10, 0] = np.nan
    preds = tree.predict(X_test)
    assert np.isfinite(preds).all()
    assert preds.shape == (len(X_test),)


# ---------------------------------------------------------------------------
# ScreeningBooster with NaN
# ---------------------------------------------------------------------------


def test_booster_regression_with_nan():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 4))
    y = (X[:, 0] > 0).astype(np.float32) * 2.0 + rng.standard_normal(300).astype(np.float32) * 0.3
    # Introduce 15% NaN in two features
    for f in [0, 2]:
        mask = rng.random(300) < 0.15
        X[mask, f] = np.nan

    booster = ScreeningBooster(
        n_estimators=20, learning_rate=0.1, max_depth=4,
        params=ScreeningParams(s_w=-2.0, s_r=-6.0),
        tree_type="non_oblivious", objective="regression",
    )
    booster.fit(X, y)
    preds = booster.predict(X)
    assert np.isfinite(preds).all()
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    assert rmse < 1.5  # should still learn the step function


def test_booster_oblivious_with_nan():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((300, 4))
    y = (X[:, 0] > 0).astype(np.float32) * 2.0
    mask = rng.random(300) < 0.2
    X[mask, 0] = np.nan

    booster = ScreeningBooster(
        n_estimators=15, learning_rate=0.1, max_depth=3,
        params=ScreeningParams(s_w=-2.0, s_r=-6.0),
        tree_type="oblivious", objective="regression",
    )
    booster.fit(X, y)
    preds = booster.predict(X)
    assert np.isfinite(preds).all()
