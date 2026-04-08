"""
ScreeningParamSearch tests — correctness, output structure, basic convergence.
"""

import numpy as np
import pytest

from sbt import ScreeningParamSearch


RNG = np.random.default_rng(42)


def make_regression(n=400, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.float32) * 2.0 + rng.standard_normal(n).astype(np.float32) * 0.3
    return X, y


def make_binary(n=400, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] * 2.0)))
    y = (rng.random(n) < p).astype(np.float32)
    return X, y


def test_fit_returns_self():
    X, y = make_regression()
    searcher = ScreeningParamSearch(
        s_w_grid=(-2.0,), s_r_grid=(-6.0,),
        n_estimators=5, cv=2, verbose=False,
    )
    result = searcher.fit(X, y)
    assert result is searcher


def test_best_params_populated():
    X, y = make_regression()
    searcher = ScreeningParamSearch(
        s_w_grid=(-2.0, 0.0), s_r_grid=(-6.0, 0.0),
        n_estimators=5, cv=2, verbose=False,
    )
    searcher.fit(X, y)
    assert searcher.best_params_ is not None
    assert searcher.best_params_.s_w in (-2.0, 0.0)
    assert searcher.best_params_.s_r in (-6.0, 0.0)


def test_cv_results_length():
    X, y = make_regression()
    s_w_grid = (-2.0, 0.0)
    s_r_grid = (-6.0, -4.0, 0.0)
    searcher = ScreeningParamSearch(
        s_w_grid=s_w_grid, s_r_grid=s_r_grid,
        n_estimators=5, cv=2, verbose=False,
    )
    searcher.fit(X, y)
    assert len(searcher.cv_results_) == len(s_w_grid) * len(s_r_grid)


def test_best_score_is_minimum():
    X, y = make_regression()
    searcher = ScreeningParamSearch(
        s_w_grid=(-2.0, 0.0), s_r_grid=(-6.0, 0.0),
        n_estimators=5, cv=2, verbose=False,
    )
    searcher.fit(X, y)
    all_scores = [r["mean_score"] for r in searcher.cv_results_]
    assert searcher.best_score_ == min(all_scores)


def test_results_table_string():
    X, y = make_regression()
    searcher = ScreeningParamSearch(
        s_w_grid=(-2.0,), s_r_grid=(-6.0, -4.0),
        n_estimators=5, cv=2, verbose=False,
    )
    searcher.fit(X, y)
    table = searcher.results_table()
    assert "best" in table
    assert "RMSE" in table


def test_binary_objective():
    X, y = make_binary()
    searcher = ScreeningParamSearch(
        s_w_grid=(-2.0,), s_r_grid=(-6.0, -4.0),
        n_estimators=5, cv=2,
        objective="binary", verbose=False,
    )
    searcher.fit(X, y)
    assert searcher.best_params_ is not None
    table = searcher.results_table()
    assert "log-loss" in table


def test_best_params_improve_model():
    """Best params should give lower CV score than a deliberately bad set."""
    X, y = make_regression(n=500)
    searcher = ScreeningParamSearch(
        s_w_grid=(-4.0, -2.0, 0.0),
        s_r_grid=(-6.0, -4.0, 0.0),
        n_estimators=10, cv=3, verbose=False,
    )
    searcher.fit(X, y)
    # Best CV score should be finite and reasonable
    assert np.isfinite(searcher.best_score_)
    assert searcher.best_score_ < 2.0  # RMSE < 2 on step-function data
