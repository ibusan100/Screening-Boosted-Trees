"""
ScreeningBooster — gradient boosting over ScreeningTree or ObliviousTree.

Each round:
  1. Compute g, h from objective loss
  2. Fit base learner on (X, g, h) using fit_gradients()
  3. y_pred += learning_rate × tree.predict(X)

Supported objectives:
  "regression"  MSE: g = y_pred − y,  h = 1
  "binary"      Log-loss: g = sigmoid(y_pred) − y,  h = p(1−p)
                Predictions in log-odds space; predict_proba() returns sigmoid.

Supported tree types:
  "non_oblivious"  ScreeningTree (M1/M2 default, per-node independent screening)
  "oblivious"      ObliviousTree (M3, per-level aggregated screening, CatBoost-style)

The Binner is fitted once on the full training set and reused every round.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .binning import Binner
from .diagnostics import ScreeningDiagnostics
from .oblivious_tree import ObliviousTree
from .screening_split import ScreeningParams
from .tree import ScreeningTree


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.clip(x.astype(np.float64), -500.0, 500.0)
    return (1.0 / (1.0 + np.exp(-x64))).astype(np.float32)


class ScreeningBooster:
    """Gradient boosting regression/classification using ScreeningTree or
    ObliviousTree as the base learner.

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
    tree_type : {"non_oblivious", "oblivious"}
        Base learner type.
    objective : {"regression", "binary"}
        Loss function.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        num_bins: int = 255,
        params: Optional[ScreeningParams] = None,
        tree_type: Literal["non_oblivious", "oblivious"] = "non_oblivious",
        objective: Literal["regression", "binary"] = "regression",
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_bins = num_bins
        self.tree_type = tree_type
        self.objective = objective

        # Boosting default: s_r=-6.0 (r≈1.0025).  Standalone-tree default s_r=0.0
        # is too strict for boosting because per-round gains — even after per-node
        # gradient normalisation — are ~10x smaller; s_r=-6.0 keeps the auto-stop
        # property while achieving competitive RMSE.
        if params is None:
            params = ScreeningParams(s_w=-2.0, s_r=-6.0, lam=1.0)
        self.params = params

        self.trees_: list = []
        self.base_score_: float = 0.0
        self.round_diagnostics_: list[ScreeningDiagnostics] = []
        self._binner: Optional[Binner] = None

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScreeningBooster":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)

        if self.objective == "regression":
            self.base_score_ = float(np.mean(y))
            y_pred = np.full(len(y), self.base_score_, dtype=np.float32)
        elif self.objective == "binary":
            p_mean = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
            self.base_score_ = float(np.log(p_mean / (1.0 - p_mean)))
            y_pred = np.full(len(y), self.base_score_, dtype=np.float32)
        else:
            raise ValueError(f"Unknown objective: {self.objective!r}")

        # Fit binner once; reuse every round.
        self._binner = Binner(self.num_bins)
        self._binner.fit(X)

        self.trees_ = []
        self.round_diagnostics_ = []

        for _ in range(self.n_estimators):
            g, h = self._gradients(y, y_pred)

            tree = self._make_tree()
            tree.fit_gradients(X, g, h, binner=self._binner)

            self.trees_.append(tree)
            self.round_diagnostics_.append(tree.diagnostics)
            y_pred = y_pred + self.learning_rate * tree.predict(X)

        return self

    def _gradients(
        self, y: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.objective == "regression":
            g = y_pred - y
            h = np.ones_like(y)
        elif self.objective == "binary":
            p = _sigmoid(y_pred)
            g = (p - y).astype(np.float32)
            h = (p * (1.0 - p)).astype(np.float32)
            h = np.clip(h, 1e-6, None)  # avoid h=0 when p→0 or p→1
        return g, h

    def _make_tree(self):
        if self.tree_type == "non_oblivious":
            return ScreeningTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                num_bins=self.num_bins,
                params=self.params,
            )
        elif self.tree_type == "oblivious":
            return ObliviousTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                num_bins=self.num_bins,
                params=self.params,
                screening_mode="per_level",
            )
        else:
            raise ValueError(f"Unknown tree_type: {self.tree_type!r}")

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw scores (log-odds for binary, values for regression)."""
        X = np.asarray(X, dtype=np.float64)
        preds = np.full(len(X), self.base_score_, dtype=np.float32)
        for tree in self.trees_:
            preds = preds + self.learning_rate * tree.predict(X)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (binary objective only)."""
        assert self.objective == "binary", "predict_proba requires objective='binary'"
        return _sigmoid(self.predict(X))

    # ------------------------------------------------------------------ #
    # Diagnostics helpers                                                  #
    # ------------------------------------------------------------------ #

    def accept_rates(self) -> np.ndarray:
        """Root/level-0 accept_rate for each boosting round. Shape: (n_estimators,)."""
        rates = []
        for diag in self.round_diagnostics_:
            r = diag.root_accept_rate
            rates.append(r if r is not None else float("nan"))
        return np.array(rates, dtype=np.float32)

    def mean_accept_rate(self) -> float:
        rates = self.accept_rates()
        valid = rates[~np.isnan(rates)]
        return float(valid.mean()) if len(valid) > 0 else float("nan")
