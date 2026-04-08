"""
ScreeningBooster — gradient boosting over ScreeningTree base learners.

Each round:
  1. g_i = y_pred_i − y_i,  h_i = 1   (MSE gradient / hessian)
  2. Fit ScreeningTree on (X, g, h) using fit_gradients()
  3. y_pred += learning_rate × tree.predict(X)

The Binner is fitted once on the full training set and reused every round to
avoid redundant quantile computation (cost is O(N log N) per feature).

Diagnostics from every round are stored in `self.round_diagnostics_` and can
be used to monitor how accept_rate evolves over training (D0 in CLAUDE.md).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .binning import Binner
from .diagnostics import ScreeningDiagnostics
from .screening_split import ScreeningParams
from .tree import ScreeningTree


class ScreeningBooster:
    """Gradient boosting regression using ScreeningTree as the base learner.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage applied to each tree's output.
    max_depth : int
        Max depth per tree.
    min_samples_leaf : int
        Min samples per leaf.
    num_bins : int
        Max quantile bins per feature.
    params : ScreeningParams or None
        Screening scalars shared across all trees.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        num_bins: int = 255,
        params: Optional[ScreeningParams] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_bins = num_bins
        # Boosting default: s_r=-6.0 (r≈1.0025).  Standalone-tree default s_r=0.0
        # is too strict for boosting because per-round gains — even after per-node
        # gradient normalisation — are ~10x smaller; s_r=-6.0 keeps the auto-stop
        # property while achieving competitive RMSE.
        if params is None:
            params = ScreeningParams(s_w=-2.0, s_r=-6.0, lam=1.0)
        self.params = params

        self.trees_: list[ScreeningTree] = []
        self.base_score_: float = 0.0
        self.round_diagnostics_: list[ScreeningDiagnostics] = []
        self._binner: Optional[Binner] = None

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScreeningBooster":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)

        self.base_score_ = float(np.mean(y))
        y_pred = np.full(len(y), self.base_score_, dtype=np.float32)

        # Fit binner once; reuse every round.
        self._binner = Binner(self.num_bins)
        self._binner.fit(X)

        self.trees_ = []
        self.round_diagnostics_ = []

        for _ in range(self.n_estimators):
            g = y_pred - y          # MSE gradient
            h = np.ones_like(y)     # MSE hessian is constant 1

            tree = ScreeningTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                num_bins=self.num_bins,
                params=self.params,
            )
            tree.fit_gradients(X, g, h, binner=self._binner)

            self.trees_.append(tree)
            self.round_diagnostics_.append(tree.diagnostics)
            y_pred = y_pred + self.learning_rate * tree.predict(X)

        return self

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        preds = np.full(len(X), self.base_score_, dtype=np.float32)
        for tree in self.trees_:
            preds = preds + self.learning_rate * tree.predict(X)
        return preds

    # ------------------------------------------------------------------ #
    # Diagnostics helpers                                                  #
    # ------------------------------------------------------------------ #

    def accept_rates(self) -> np.ndarray:
        """Root accept_rate for each boosting round. Shape: (n_estimators,)."""
        rates = []
        for diag in self.round_diagnostics_:
            r = diag.root_accept_rate
            rates.append(r if r is not None else float("nan"))
        return np.array(rates, dtype=np.float32)

    def mean_accept_rate(self) -> float:
        rates = self.accept_rates()
        valid = rates[~np.isnan(rates)]
        return float(valid.mean()) if len(valid) > 0 else float("nan")
