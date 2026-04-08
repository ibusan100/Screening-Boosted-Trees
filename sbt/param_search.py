"""
ScreeningParamSearch — grid search for (s_w, s_r) via cross-validated loss.

Design (CLAUDE.md § M4 / D1 Phase 2):
- No kernel changes required: (s_w, s_r) enter only post-histogram.
- Uses K-fold CV so small datasets get reliable estimates.
- Metric: RMSE for regression, log-loss for binary (both "lower is better").
- Default grids cover the empirically calibrated ranges:
    s_w ∈ {-4, -2, 0}    (gain temperature; -2 is M1/M2 default)
    s_r ∈ {-6, -4, -2, 0} (-6=boosting regression; -4=classification)
"""

from __future__ import annotations

import itertools
from typing import Literal, Optional

import numpy as np

from .screening_split import ScreeningParams


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# ScreeningParamSearch
# ---------------------------------------------------------------------------


class ScreeningParamSearch:
    """K-fold grid search over ScreeningParams (s_w, s_r).

    Finds the (s_w, s_r) pair that minimises the cross-validated metric
    (RMSE for regression, log-loss for binary classification).

    Parameters
    ----------
    s_w_grid : sequence of float
        Candidates for s_w (gain temperature, log-scale).
    s_r_grid : sequence of float
        Candidates for s_r (acceptance width, log-scale).
    n_estimators : int
        Boosting rounds used in each CV fold (kept small for speed).
    learning_rate : float
    max_depth : int
    min_samples_leaf : int
    lam : float
        L2 regularisation (held fixed during search).
    cv : int
        Number of cross-validation folds.
    objective : {"regression", "binary"}
    tree_type : {"non_oblivious", "oblivious"}
    verbose : bool
        Print per-candidate scores.
    """

    # Default grids cover the calibrated ranges from M2/M3 experiments.
    _DEFAULT_S_W = (-4.0, -2.0, 0.0)
    _DEFAULT_S_R = (-6.0, -4.0, -2.0, 0.0)

    def __init__(
        self,
        s_w_grid: tuple[float, ...] = _DEFAULT_S_W,
        s_r_grid: tuple[float, ...] = _DEFAULT_S_R,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        lam: float = 1.0,
        cv: int = 3,
        objective: Literal["regression", "binary"] = "regression",
        tree_type: Literal["non_oblivious", "oblivious"] = "oblivious",
        verbose: bool = True,
    ):
        self.s_w_grid = tuple(s_w_grid)
        self.s_r_grid = tuple(s_r_grid)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lam = lam
        self.cv = cv
        self.objective = objective
        self.tree_type = tree_type
        self.verbose = verbose

        self.best_params_: Optional[ScreeningParams] = None
        self.best_score_: float = float("inf")
        self.cv_results_: list[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScreeningParamSearch":
        """Run grid search and store best_params_."""
        from .booster import ScreeningBooster  # local import to avoid circular

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)
        n = len(X)

        fold_idx = self._make_folds(n)

        self.cv_results_ = []
        self.best_score_ = float("inf")
        self.best_params_ = None

        candidates = list(itertools.product(self.s_w_grid, self.s_r_grid))
        n_cand = len(candidates)

        for ci, (s_w, s_r) in enumerate(candidates):
            params = ScreeningParams(s_w=s_w, s_r=s_r, lam=self.lam)
            fold_scores = []

            for train_idx, val_idx in fold_idx:
                model = ScreeningBooster(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    params=params,
                    tree_type=self.tree_type,
                    objective=self.objective,
                )
                model.fit(X[train_idx], y[train_idx])

                if self.objective == "regression":
                    preds = model.predict(X[val_idx])
                    score = _rmse(y[val_idx], preds)
                else:
                    probs = model.predict_proba(X[val_idx])
                    score = _log_loss(y[val_idx], probs)

                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            std_score  = float(np.std(fold_scores))
            self.cv_results_.append({
                "s_w": s_w, "s_r": s_r,
                "mean_score": mean_score, "std_score": std_score,
            })

            if self.verbose:
                metric = "RMSE" if self.objective == "regression" else "log-loss"
                print(f"  [{ci+1:2d}/{n_cand}] s_w={s_w:5.1f}, s_r={s_r:5.1f} → "
                      f"{metric}={mean_score:.4f}±{std_score:.4f}")

            if mean_score < self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params

        if self.verbose:
            metric = "RMSE" if self.objective == "regression" else "log-loss"
            print(f"\n  Best: s_w={self.best_params_.s_w}, s_r={self.best_params_.s_r}"
                  f"  {metric}={self.best_score_:.4f}")
        return self

    def _make_folds(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (train_idx, val_idx) for K-fold."""
        idx = np.arange(n)
        fold_size = n // self.cv
        folds = []
        for k in range(self.cv):
            val_start = k * fold_size
            val_end = val_start + fold_size if k < self.cv - 1 else n
            val_idx = idx[val_start:val_end]
            train_idx = np.concatenate([idx[:val_start], idx[val_end:]])
            folds.append((train_idx, val_idx))
        return folds

    def results_table(self) -> str:
        """Return a formatted string of all CV results sorted by score."""
        metric = "RMSE" if self.objective == "regression" else "log-loss"
        rows = sorted(self.cv_results_, key=lambda r: r["mean_score"])
        lines = [f"{'s_w':>6} {'s_r':>6}  {metric:>10} ± std"]
        lines.append("-" * 36)
        for r in rows:
            marker = " ← best" if (r["s_w"] == self.best_params_.s_w and
                                    r["s_r"] == self.best_params_.s_r) else ""
            lines.append(f"{r['s_w']:6.1f} {r['s_r']:6.1f}  "
                         f"{r['mean_score']:10.4f} ± {r['std_score']:.4f}{marker}")
        return "\n".join(lines)
