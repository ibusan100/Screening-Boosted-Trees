"""
ScreeningTree — single regression decision tree using screening-based split
selection.

Design (CLAUDE.md § M1):
- Non-oblivious: each node independently chooses its best (feature, bin).
- Split criterion: screening transform applied to MSE gain.
- Stopping: max_depth, min_samples_leaf, or screening rejects all splits.

Two fitting modes:
  fit(X, y)             — standalone tree; gradients re-centred per node
                          (g_i = y_i − mean(y_node)); leaf value = mean(y_node).
  fit_gradients(X, g, h)— boosting mode; g/h are global residuals supplied by
                          the caller; leaf value = −G/(H+λ) (Newton step).
"""

from __future__ import annotations

from collections import deque
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
# Internal node representation
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    node_id: int
    depth: int
    sample_idx: np.ndarray

    is_leaf: bool = False
    leaf_value: float = 0.0

    split_feature: int = -1
    split_bin: int = -1
    split_threshold: float = float("nan")
    left_child: int = -1
    right_child: int = -1


# ---------------------------------------------------------------------------
# ScreeningTree
# ---------------------------------------------------------------------------


class ScreeningTree:
    """Single regression tree with screening-based split selection.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth (root is depth 0).
    min_samples_leaf : int
        Minimum samples in each child after a split.
    num_bins : int
        Maximum quantile bins per feature (passed to Binner).
    params : ScreeningParams or None
        Screening scalars. None → ScreeningParams() defaults.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        num_bins: int = 255,
        params: Optional[ScreeningParams] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_bins = num_bins
        self.params = params if params is not None else ScreeningParams()

        self._binner: Optional[Binner] = None
        self._nodes: list[_Node] = []
        self._diagnostics: Optional[ScreeningDiagnostics] = None

    # ------------------------------------------------------------------ #
    # Public fit interfaces                                                #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScreeningTree":
        """Standalone tree: gradients recentred per node (g_i = y_i − μ_node)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)
        self._binner = Binner(self.num_bins)
        X_binned = self._binner.fit_transform(X).astype(np.int32)

        def g_fn(idx: np.ndarray):
            y_n = y[idx]
            mu = float(np.mean(y_n))
            g = (y_n - mu).astype(np.float32)
            h = np.ones(len(idx), dtype=np.float32)
            return g, h

        def leaf_fn(idx: np.ndarray, _g, _h) -> float:
            return float(np.mean(y[idx]))

        self._bfs_core(X_binned, g_fn, leaf_fn)
        return self

    def fit_gradients(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        binner: Optional[Binner] = None,
    ) -> "ScreeningTree":
        """Boosting mode: g/h supplied externally; leaf = −G/(H+λ).

        Parameters
        ----------
        X       : (N, F) float — original features.
        g       : (N,)   float32 — first-order gradient from the loss.
        h       : (N,)   float32 — second-order gradient (hessian).
        binner  : pre-fitted Binner to reuse (avoids re-fitting quantiles per round).
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
            # Normalise g and h per node so norm_gain ∈ [0, 1] for any loss.
            # g: centre (G_total=0 → parent term cancels) then divide by std
            #    → norm_gain = within-node variance fraction explained.
            # h: divide by mean (H_total → n_node) so the scale of h_i
            #    (which varies across objectives, e.g. p(1-p) for binary)
            #    does not affect the threshold.  Both normalisations happen
            #    only for split scoring; leaf value uses the ORIGINAL g/h.
            g_n = g[idx]
            g_centred = g_n - g_n.mean()
            g_std = float(np.std(g_centred)) + 1e-8
            g_norm = (g_centred / g_std).astype(np.float32)

            h_n = h[idx]
            h_mean = float(h_n.mean()) + 1e-8
            h_norm = (h_n / h_mean).astype(np.float32)
            return g_norm, h_norm

        def leaf_fn(idx: np.ndarray, _g_centred: np.ndarray, _h: np.ndarray) -> float:
            return float(-g[idx].sum() / (h[idx].sum() + lam))

        self._bfs_core(X_binned, g_fn, leaf_fn)
        return self

    # ------------------------------------------------------------------ #
    # Shared BFS core                                                      #
    # ------------------------------------------------------------------ #

    def _bfs_core(
        self,
        X_binned: np.ndarray,
        g_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        leaf_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> None:
        N = X_binned.shape[0]
        actual_max_bins = self._binner.max_bins()
        F = X_binned.shape[1]

        self._nodes = []
        self._diagnostics = ScreeningDiagnostics()

        root = _Node(node_id=0, depth=0, sample_idx=np.arange(N, dtype=np.int32))
        queue: deque[_Node] = deque([root])
        self._nodes.append(root)
        next_id = 1

        while queue:
            node = queue.popleft()
            idx = node.sample_idx
            n = len(idx)

            g_n, h_n = g_fn(idx)
            node.leaf_value = leaf_fn(idx, g_n, h_n)

            # Stopping: depth or size limit → leaf, no diagnostics entry needed
            if node.depth >= self.max_depth or n < 2 * self.min_samples_leaf:
                node.is_leaf = True
                self._diagnostics.nodes.append(NodeDiagnostics(
                    node_id=node.node_id, depth=node.depth, n_samples=n,
                    n_candidates=0, n_accepted=0, accept_rate=0.0,
                    rho_max=0.0, rho_mean=float("nan"), rho_median=float("nan"),
                    split_found=False,
                ))
                continue

            # Build histogram (single node)
            nid = np.zeros(n, dtype=np.int32)
            hist_G, hist_H = build_histogram_numpy(
                X_binned[idx], g_n, h_n, nid,
                num_nodes=1, num_bins=actual_max_bins,
            )

            result = screening_split_numpy(hist_G, hist_H, self.params)

            # Record diagnostics
            rho_flat = result["rho"].ravel()
            accepted_vals = rho_flat[rho_flat > 0.0]
            n_cand = int((actual_max_bins - 1) * F)
            n_acc = len(accepted_vals)
            self._diagnostics.nodes.append(NodeDiagnostics(
                node_id=node.node_id, depth=node.depth, n_samples=n,
                n_candidates=n_cand, n_accepted=n_acc,
                accept_rate=n_acc / n_cand if n_cand > 0 else 0.0,
                rho_max=float(rho_flat.max()),
                rho_mean=float(accepted_vals.mean()) if n_acc > 0 else float("nan"),
                rho_median=float(np.median(accepted_vals)) if n_acc > 0 else float("nan"),
                split_found=bool(result["accepted_mask"][0]),
            ))

            if not result["accepted_mask"][0]:
                node.is_leaf = True
                continue

            best_feat = int(result["best_feat"][0])
            best_bin  = int(result["best_bin"][0])

            left_mask  = X_binned[idx, best_feat] <= best_bin
            right_mask = ~left_mask

            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                node.is_leaf = True
                continue

            node.split_feature   = best_feat
            node.split_bin       = best_bin
            node.split_threshold = self._binner.threshold(best_feat, best_bin)

            left_node  = _Node(node_id=next_id,     depth=node.depth + 1, sample_idx=idx[left_mask])
            right_node = _Node(node_id=next_id + 1, depth=node.depth + 1, sample_idx=idx[right_mask])
            next_id += 2

            node.left_child  = left_node.node_id
            node.right_child = right_node.node_id
            self._nodes += [left_node, right_node]
            queue += [left_node, right_node]

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._binner is not None, "Call fit() or fit_gradients() first."
        X = np.asarray(X, dtype=np.float64)
        node_map = {n.node_id: n for n in self._nodes}
        root = node_map[0]
        preds = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            node = root
            while not node.is_leaf:
                if X[i, node.split_feature] <= node.split_threshold:
                    node = node_map[node.left_child]
                else:
                    node = node_map[node.right_child]
            preds[i] = node.leaf_value
        return preds

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def diagnostics(self) -> Optional[ScreeningDiagnostics]:
        return self._diagnostics

    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self._nodes if n.is_leaf)

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    def depth(self) -> int:
        return max(n.depth for n in self._nodes) if self._nodes else 0
