"""
ObliviousTree — oblivious regression/gradient tree using screening-based
per-level split selection.

Design (CLAUDE.md § M3 / D2):
- At each depth d, ALL nodes at that level share the SINGLE best (feature, bin)
  determined by screening (CatBoost-style oblivious structure).
- screening_mode="per_level" (default): aggregate gradient histograms across
  all nodes at depth d, run screening once, use the result for every node at
  that level.  Stop the moment max(rho) == 0 across the aggregated histogram.
- Leaf layout: 2^depth contiguous values indexed by the bit-path
  (0=left, 1=right at each level) from root to leaf.
- Prediction is O(depth) with a fully vectorised leaf-index lookup.

Two fitting modes (same interface as ScreeningTree):
  fit(X, y)              — standalone; gradients recentred + std-normalised per node.
  fit_gradients(X, g, h) — boosting; g normalised per node (mean=0, std=1) for split
                           scoring; leaf value = −G_orig / (H_orig + λ) Newton step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .binning import Binner
from .diagnostics import NodeDiagnostics, ScreeningDiagnostics
from .screening_split import (
    ScreeningParams,
    build_histogram_numpy,
    screening_split_numpy,
)


# ---------------------------------------------------------------------------
# Internal per-level split record
# ---------------------------------------------------------------------------


@dataclass
class _LevelSplit:
    depth: int
    feat: int
    bin_idx: int
    threshold: float


# ---------------------------------------------------------------------------
# ObliviousTree
# ---------------------------------------------------------------------------


class ObliviousTree:
    """Single oblivious tree with per-level screening-based split selection.

    Parameters
    ----------
    max_depth : int
        Maximum depth.  The tree has at most 2**max_depth leaves.
    min_samples_leaf : int
        Nodes with fewer than 2×min_samples_leaf samples are not considered
        when searching for the level split (but are still split passively using
        the level's chosen (feat, bin)).
    num_bins : int
        Maximum quantile bins per feature (passed to Binner).
    params : ScreeningParams or None
        Screening scalars.  None → ScreeningParams() defaults.
    screening_mode : {"per_level"}
        How to select the split at each level.
        "per_level": aggregate histograms across all nodes at depth d, screen
        once, use the winner for every node.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        num_bins: int = 255,
        params: Optional[ScreeningParams] = None,
        screening_mode: str = "per_level",
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_bins = num_bins
        self.params = params if params is not None else ScreeningParams()
        self.screening_mode = screening_mode

        self._binner: Optional[Binner] = None
        self._level_splits: list[_LevelSplit] = []
        self._leaf_values: Optional[np.ndarray] = None
        self._diagnostics: Optional[ScreeningDiagnostics] = None

    # ------------------------------------------------------------------ #
    # Public fit interfaces                                                #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ObliviousTree":
        """Standalone tree: gradients recentred + normalised per node."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)
        self._binner = Binner(self.num_bins)
        X_binned = self._binner.fit_transform(X).astype(np.int32)

        def g_fn(idx: np.ndarray):
            y_n = y[idx]
            g = (y_n - y_n.mean()).astype(np.float32)
            g_std = float(np.std(g)) + 1e-8
            return (g / g_std).astype(np.float32), np.ones(len(idx), dtype=np.float32)

        def leaf_fn(idx: np.ndarray) -> float:
            return float(np.mean(y[idx])) if len(idx) > 0 else 0.0

        self._fit_core(X_binned, g_fn, leaf_fn)
        return self

    def fit_gradients(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        binner: Optional[Binner] = None,
    ) -> "ObliviousTree":
        """Boosting mode: g/h supplied externally; leaf = −G/(H+λ).

        Parameters
        ----------
        X      : (N, F) float64
        g      : (N,)   float32 — first-order gradient from the loss.
        h      : (N,)   float32 — second-order gradient (hessian).
        binner : pre-fitted Binner to reuse across rounds.
        """
        X = np.asarray(X, dtype=np.float64)
        g = np.asarray(g, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)

        if binner is not None:
            self._binner = binner
            X_binned = binner.transform(X).astype(np.int32)
        else:
            self._binner = Binner(self.num_bins)
            X_binned = self._binner.fit_transform(X).astype(np.int32)

        lam = self.params.lam

        def g_fn(idx: np.ndarray):
            g_n = g[idx]
            g_c = g_n - g_n.mean()
            g_std = float(np.std(g_c)) + 1e-8
            return (g_c / g_std).astype(np.float32), h[idx]

        def leaf_fn(idx: np.ndarray) -> float:
            if len(idx) == 0:
                return 0.0
            return float(-g[idx].sum() / (h[idx].sum() + lam))

        self._fit_core(X_binned, g_fn, leaf_fn)
        return self

    # ------------------------------------------------------------------ #
    # Core level-by-level fit                                              #
    # ------------------------------------------------------------------ #

    def _fit_core(
        self,
        X_binned: np.ndarray,
        g_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        leaf_fn: Callable[[np.ndarray], float],
    ) -> None:
        N = X_binned.shape[0]
        B = self._binner.max_bins()
        F = X_binned.shape[1]

        self._level_splits = []
        self._diagnostics = ScreeningDiagnostics()

        # nodes: list of index arrays; starts as [all_N], doubles each level.
        nodes: list[np.ndarray] = [np.arange(N, dtype=np.int32)]

        for depth in range(self.max_depth):
            # Nodes with enough samples to participate in split search
            splittable = [idx for idx in nodes if len(idx) >= 2 * self.min_samples_leaf]

            if not splittable:
                break  # Every node is too small — stop growing

            feat, bin_idx, accepted = self._find_level_split(
                X_binned, g_fn, splittable, B, F, depth
            )

            if not accepted:
                break  # Screening rejected all level candidates

            threshold = self._binner.threshold(feat, bin_idx)
            self._level_splits.append(_LevelSplit(depth, feat, bin_idx, threshold))

            # Split every node (including small ones — they're split passively)
            new_nodes: list[np.ndarray] = []
            for idx in nodes:
                mask = X_binned[idx, feat] <= bin_idx
                new_nodes.append(idx[mask])
                new_nodes.append(idx[~mask])
            nodes = new_nodes

        # Leaf values indexed by bit-path (see predict)
        self._leaf_values = np.array(
            [leaf_fn(idx) for idx in nodes], dtype=np.float32
        )

    def _find_level_split(
        self,
        X_binned: np.ndarray,
        g_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        splittable: list[np.ndarray],
        num_bins: int,
        F: int,
        depth: int,
    ) -> tuple[int, int, bool]:
        """Aggregate histograms across splittable nodes and apply screening once."""
        agg_G = np.zeros((1, F, num_bins), dtype=np.float32)
        agg_H = np.zeros((1, F, num_bins), dtype=np.float32)
        total_n = 0

        for idx in splittable:
            g_norm, h_n = g_fn(idx)
            nid = np.zeros(len(idx), dtype=np.int32)
            hG, hH = build_histogram_numpy(X_binned[idx], g_norm, h_n, nid, 1, num_bins)
            agg_G += hG
            agg_H += hH
            total_n += len(idx)

        result = screening_split_numpy(agg_G, agg_H, self.params)

        # Diagnostics — one entry per level (node_id = depth for clarity)
        rho_flat = result["rho"].ravel()
        accepted_vals = rho_flat[rho_flat > 0.0]
        n_cand = int((num_bins - 1) * F)
        n_acc = len(accepted_vals)
        self._diagnostics.nodes.append(NodeDiagnostics(
            node_id=depth, depth=depth, n_samples=total_n,
            n_candidates=n_cand, n_accepted=n_acc,
            accept_rate=n_acc / n_cand if n_cand > 0 else 0.0,
            rho_max=float(rho_flat.max()),
            rho_mean=float(accepted_vals.mean()) if n_acc > 0 else float("nan"),
            rho_median=float(np.median(accepted_vals)) if n_acc > 0 else float("nan"),
            split_found=bool(result["accepted_mask"][0]),
        ))

        if not result["accepted_mask"][0]:
            return -1, -1, False

        return int(result["best_feat"][0]), int(result["best_bin"][0]), True

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorised O(depth) prediction via bit-path leaf indexing."""
        assert self._binner is not None, "Call fit() or fit_gradients() first."
        X = np.asarray(X, dtype=np.float64)

        if not self._level_splits:
            # No splits — return single leaf value for all
            return np.full(len(X), self._leaf_values[0], dtype=np.float32)

        leaf_idx = np.zeros(len(X), dtype=np.int32)
        for split in self._level_splits:
            goes_right = (X[:, split.feat] > split.threshold).astype(np.int32)
            leaf_idx = leaf_idx * 2 + goes_right

        return self._leaf_values[leaf_idx]

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def diagnostics(self) -> Optional[ScreeningDiagnostics]:
        return self._diagnostics

    @property
    def depth(self) -> int:
        return len(self._level_splits)

    @property
    def n_leaves(self) -> int:
        return len(self._leaf_values) if self._leaf_values is not None else 0

    @property
    def n_nodes(self) -> int:
        """Total nodes in the complete binary tree (2^(depth+1) - 1)."""
        d = self.depth
        return (2 ** (d + 1)) - 1 if d > 0 else 1
