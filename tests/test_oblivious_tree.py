"""
ObliviousTree tests — correctness, screening semantics, booster integration.
"""

import numpy as np
import pytest

from ibu_boost import ObliviousTree, ScreeningBooster, ScreeningParams


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_regression(n=500, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.float32) * 2.0 + rng.standard_normal(n).astype(np.float32) * 0.3
    return X, y


def make_binary(n=500, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] * 2.0)))
    y = (rng.random(n) < p).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# ObliviousTree unit tests
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    X, y = make_regression()
    tree = ObliviousTree(max_depth=3, min_samples_leaf=10)
    result = tree.fit(X, y)
    assert result is tree


def test_predict_shape():
    X, y = make_regression()
    tree = ObliviousTree(max_depth=3).fit(X, y)
    preds = tree.predict(X)
    assert preds.shape == (len(X),)
    assert preds.dtype == np.float32


def test_leaf_count_at_most_2_pow_depth():
    X, y = make_regression(n=2000)
    for max_depth in [1, 2, 4, 6]:
        tree = ObliviousTree(max_depth=max_depth, min_samples_leaf=10).fit(X, y)
        assert tree.n_leaves <= 2 ** max_depth


def test_depth_leq_max_depth():
    X, y = make_regression(n=2000)
    tree = ObliviousTree(max_depth=4, min_samples_leaf=10).fit(X, y)
    assert tree.depth <= 4


def test_all_nodes_same_features():
    """All level-splits should use different features or same — just check structure."""
    X, y = make_regression(n=2000)
    tree = ObliviousTree(max_depth=4, min_samples_leaf=10).fit(X, y)
    assert len(tree._level_splits) == tree.depth
    for split in tree._level_splits:
        assert 0 <= split.feat < X.shape[1]


def test_step_function_learned():
    """Clear step function: tree should fit it with low RMSE."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((1000, 4)).astype(np.float64)
    y = np.where(X[:, 0] > 0.0, 1.0, -1.0).astype(np.float32)
    tree = ObliviousTree(
        max_depth=4, min_samples_leaf=5,
        params=ScreeningParams(s_w=-2.0, s_r=0.0, lam=0.1),
    ).fit(X, y)
    rmse = float(np.sqrt(np.mean((tree.predict(X) - y) ** 2)))
    assert rmse < 0.5, f"RMSE={rmse:.4f} too large for step function"


def test_screening_stops_on_noise():
    """Pure noise: screening should reject root (or make a shallow tree)."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((500, 4)).astype(np.float64)
    y = rng.standard_normal(500).astype(np.float32)
    tree = ObliviousTree(
        max_depth=6, min_samples_leaf=10,
        params=ScreeningParams(s_w=-2.0, s_r=0.0, lam=1.0),
    ).fit(X, y)
    # Noisy data: expect ≤1 level split (screening should kill most)
    assert tree.depth <= 2, f"depth={tree.depth} on pure noise"


def test_fit_gradients_basic():
    X, y = make_regression(n=1000)
    g = np.random.default_rng(0).standard_normal(len(y)).astype(np.float32)
    h = np.ones(len(y), dtype=np.float32)
    tree = ObliviousTree(max_depth=3, min_samples_leaf=10).fit_gradients(X, g, h)
    preds = tree.predict(X)
    assert preds.shape == (len(X),)


def test_diagnostics_populated():
    X, y = make_regression(n=1000)
    tree = ObliviousTree(max_depth=4, min_samples_leaf=10).fit(X, y)
    diag = tree.diagnostics
    assert diag is not None
    # Each entry corresponds to one level
    assert len(diag.nodes) == tree.depth or len(diag.nodes) == tree.depth + 1


def test_predict_no_splits():
    """Tree with max_depth=0 effectively — all screening rejected."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 2)).astype(np.float64)
    y = rng.standard_normal(200).astype(np.float32)
    # Crank s_r to 100 to force rejection of everything
    tree = ObliviousTree(
        max_depth=6, min_samples_leaf=5,
        params=ScreeningParams(s_w=10.0, s_r=10.0, lam=1.0),
    ).fit(X, y)
    assert tree.depth == 0
    preds = tree.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(preds == preds[0])  # All same leaf value


# ---------------------------------------------------------------------------
# ScreeningBooster integration tests
# ---------------------------------------------------------------------------


def test_booster_oblivious_regression():
    X, y = make_regression(n=1000)
    model = ScreeningBooster(
        n_estimators=20, learning_rate=0.1, max_depth=4,
        tree_type="oblivious",
        params=ScreeningParams(s_w=-2.0, s_r=-6.0, lam=1.0),
    )
    model.fit(X, y)
    preds = model.predict(X)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    assert rmse < 0.5, f"RMSE={rmse:.4f}"


def test_booster_binary_classification():
    X, y = make_binary(n=800)
    model = ScreeningBooster(
        n_estimators=30, learning_rate=0.1, max_depth=4,
        objective="binary",
        params=ScreeningParams(s_w=-2.0, s_r=-6.0, lam=1.0),
    )
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))
    # Should achieve > 70% accuracy on a clear signal
    acc = float(np.mean((probs > 0.5) == y))
    assert acc > 0.70, f"Accuracy={acc:.3f}"


def test_booster_oblivious_binary():
    X, y = make_binary(n=800)
    model = ScreeningBooster(
        n_estimators=30, learning_rate=0.1, max_depth=4,
        tree_type="oblivious", objective="binary",
        params=ScreeningParams(s_w=-2.0, s_r=-6.0, lam=1.0),
    )
    model.fit(X, y)
    probs = model.predict_proba(X)
    acc = float(np.mean((probs > 0.5) == y))
    assert acc > 0.70, f"Accuracy={acc:.3f}"


def test_predict_proba_requires_binary():
    X, y = make_regression()
    model = ScreeningBooster(n_estimators=5, objective="regression")
    model.fit(X, y)
    with pytest.raises(AssertionError):
        model.predict_proba(X)
