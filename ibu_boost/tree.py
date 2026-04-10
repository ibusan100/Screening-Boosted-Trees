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
    build_missing_stats,
    screening_split_numpy,
)

_TRITON_AVAILABLE: bool | None = None   # cached after first check


def _triton_available() -> bool:
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import torch
            _TRITON_AVAILABLE = torch.cuda.is_available()
        except ImportError:
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


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
    default_dir: int = 0  # 0=missing→left, 1=missing→right
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
        device: str = "cpu",
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_bins = num_bins
        self.params = params if params is not None else ScreeningParams()
        self._device = device

        self._binner: Optional[Binner] = None
        self._nodes: list[_Node] = []
        self._diagnostics: Optional[ScreeningDiagnostics] = None
        self._X_gpu = None   # pre-loaded CUDA tensor (int32 binned X)
        self._g_gpu = None   # pre-loaded CUDA tensor (float32 raw gradients)
        self._h_gpu = None   # pre-loaded CUDA tensor (float32 raw hessians)
        self._y_gpu = None   # pre-loaded CUDA tensor (float32 targets, standalone)
        self._mode: str = "standalone"   # "standalone" | "boosting"

    # ------------------------------------------------------------------ #
    # Public fit interfaces                                                #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        binner: Optional[Binner] = None,
        X_binned: Optional[np.ndarray] = None,
        X_gpu=None,
    ) -> "ScreeningTree":
        """Standalone tree: gradients recentred per node (g_i = y_i − μ_node)."""
        y = np.asarray(y, dtype=np.float32)
        self._mode = "standalone"

        if X_binned is not None:
            self._binner = binner
        else:
            X = np.asarray(X, dtype=np.float64)
            self._binner = Binner(self.num_bins)
            X_binned = self._binner.fit_transform(X).astype(np.int32)

        if self._device == "cuda":
            import torch
            self._y_gpu = torch.from_numpy(y).cuda()
            self._g_gpu = None
            self._h_gpu = None

        def g_fn(idx: np.ndarray):
            y_n = y[idx]
            mu = float(np.mean(y_n))
            g = (y_n - mu).astype(np.float32)
            h = np.ones(len(idx), dtype=np.float32)
            return g, h

        def leaf_fn(idx: np.ndarray, _g, _h) -> float:
            return float(np.mean(y[idx]))

        self._maybe_upload_X(X_binned, X_gpu=X_gpu)
        self._bfs_core(X_binned, g_fn, leaf_fn)
        return self

    def fit_gradients(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        binner: Optional[Binner] = None,
        *,
        X_binned: Optional[np.ndarray] = None,
        X_gpu=None,
        g_gpu=None,
        h_gpu=None,
    ) -> "ScreeningTree":
        """Boosting mode: g/h supplied externally; leaf = −G/(H+λ).

        Parameters
        ----------
        X       : (N, F) float — original features.
        g       : (N,)   float32 — first-order gradient from the loss.
        h       : (N,)   float32 — second-order gradient (hessian).
        binner  : pre-fitted Binner to reuse (avoids re-fitting quantiles per round).
        X_binned: optional pre-computed binned X (skips binner.transform).
        X_gpu   : optional pre-loaded GPU tensor for X_binned (skips re-upload).
        g_gpu   : optional pre-loaded GPU tensor for g (skips per-round upload).
        h_gpu   : optional pre-loaded GPU tensor for h.
        """
        g = np.asarray(g, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)
        self._mode = "boosting"

        if X_binned is not None:
            self._binner = binner
        elif binner is not None:
            X = np.asarray(X, dtype=np.float64)
            self._binner = binner
            X_binned = binner.transform(X).astype(np.int32)
        else:
            X = np.asarray(X, dtype=np.float64)
            self._binner = Binner(self.num_bins)
            X_binned = self._binner.fit_transform(X).astype(np.int32)

        lam = self.params.lam

        if self._device == "cuda":
            if g_gpu is not None:
                self._g_gpu = g_gpu
                self._h_gpu = h_gpu
            else:
                import torch
                self._g_gpu = torch.from_numpy(g).cuda()
                self._h_gpu = torch.from_numpy(h).cuda()
            self._y_gpu = None

        def g_fn(idx: np.ndarray):
            # CPU normalisation — only used in CPU mode (_bfs_core skips this
            # in CUDA mode and lets _screen_batch normalise on GPU instead).
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

        self._maybe_upload_X(X_binned, X_gpu=X_gpu)
        self._bfs_core(X_binned, g_fn, leaf_fn)
        return self

    def _maybe_upload_X(self, X_binned: np.ndarray, X_gpu=None) -> None:
        """Upload X_binned to GPU once at fit start (device='cuda' only).

        If X_gpu is provided (pre-loaded by caller), reuse it to avoid
        redundant round-to-round uploads.
        """
        if self._device == "cuda":
            if X_gpu is not None:
                self._X_gpu = X_gpu
            else:
                import torch
                self._X_gpu = torch.from_numpy(X_binned).cuda()
        else:
            self._X_gpu = None

    # ------------------------------------------------------------------ #
    # Batched histogram + screening dispatch                               #
    # ------------------------------------------------------------------ #

    def _screen_batch(
        self,
        batch_items: list[tuple[np.ndarray, object, object]],
        X_binned: np.ndarray,
        num_bins: int,
    ) -> list[dict]:
        """Build histograms and run screening for a batch of nodes in one shot.

        Parameters
        ----------
        batch_items : list of (idx [n], g_n_or_None, h_n_or_None)
            In CUDA mode g_n/h_n are None (GPU gather+normalise instead).
            In CPU mode g_n/h_n are pre-normalised float32 arrays.
        X_binned    : (N, F) int32 — binned feature matrix (CPU, for CPU path).
        num_bins    : int

        Returns
        -------
        list of result dicts (one per item), each with
        best_feat, best_bin, best_rho, accepted_mask, best_default_dir.
        """
        K = len(batch_items)
        if K == 0:
            return []

        if self._device == "cuda":
            import torch
            from .kernels.screening_split_triton import (
                build_histogram_triton,
                build_missing_stats_triton,
                normalize_gh_batched_gpu,
                normalize_y_batched_gpu,
                screening_split_triton,
            )
            # Gather sample indices only; X/g/h come from pre-loaded GPU tensors.
            all_idx = np.concatenate([it[0] for it in batch_items])
            all_nid = np.concatenate([
                np.full(len(it[0]), i, dtype=np.int32) for i, it in enumerate(batch_items)
            ])
            idx_t = torch.from_numpy(all_idx).long().cuda()
            nid_t = torch.from_numpy(all_nid).long().cuda()

            X_t = self._X_gpu[idx_t]            # GPU gather — no CPU X copy

            if self._mode == "boosting":
                g_raw = self._g_gpu[idx_t]       # GPU gather — no CPU g copy
                h_raw = self._h_gpu[idx_t]
                g_t, h_t = normalize_gh_batched_gpu(g_raw, h_raw, nid_t, K)
            else:  # standalone
                y_raw = self._y_gpu[idx_t]
                g_t, h_t = normalize_y_batched_gpu(y_raw, nid_t, K, std_normalize=False)

            hG, hH = build_histogram_triton(X_t, g_t, h_t, nid_t.int(), K, num_bins)
            Gm, Hm = build_missing_stats_triton(X_t, g_t, h_t, nid_t.int(), K)
            out    = screening_split_triton(hG, hH, self.params, Gm, Hm)

            return [{
                "best_feat":        out["best_feat"][i:i+1].cpu().numpy(),
                "best_bin":         out["best_bin"][i:i+1].cpu().numpy(),
                "best_rho":         out["best_rho"][i:i+1].cpu().numpy(),
                "accepted_mask":    out["accepted_mask"][i:i+1].cpu().numpy(),
                "best_default_dir": out["best_default_dir"][i:i+1].cpu().numpy(),
            } for i in range(K)]
        else:
            results = []
            for idx, g_n, h_n in batch_items:
                X_node = X_binned[idx]
                nid = np.zeros(len(g_n), dtype=np.int32)
                hG, hH = build_histogram_numpy(X_node, g_n, h_n, nid, 1, num_bins)
                Gm, Hm = build_missing_stats(X_node, g_n, h_n, nid, 1)
                results.append(screening_split_numpy(hG, hH, self.params, Gm, Hm))
            return results

    # ------------------------------------------------------------------ #
    # BFS core — batched across the whole queue frontier                   #
    # ------------------------------------------------------------------ #

    def _bfs_core(
        self,
        X_binned: np.ndarray,
        g_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        leaf_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> None:
        N   = X_binned.shape[0]
        B   = self._binner.max_bins()
        F   = X_binned.shape[1]
        cuda = self._device == "cuda"

        self._nodes = []
        self._diagnostics = ScreeningDiagnostics()

        root = _Node(node_id=0, depth=0, sample_idx=np.arange(N, dtype=np.int32))
        queue: deque[_Node] = deque([root])
        self._nodes.append(root)
        next_id = 1

        while queue:
            # ---- Phase 1: drain entire queue; compute leaf values --------
            batch_nodes = list(queue)
            queue.clear()

            splittable: list[tuple[_Node, np.ndarray, object, object]] = []
            for node in batch_nodes:
                idx = node.sample_idx
                n   = len(idx)

                # In CUDA mode skip CPU g_fn — GPU normalisation happens in
                # _screen_batch.  leaf_fn ignores g_n/h_n (uses closure).
                if cuda:
                    g_n, h_n = None, None
                else:
                    g_n, h_n = g_fn(idx)

                node.leaf_value = leaf_fn(idx, g_n, h_n)

                if node.depth >= self.max_depth or n < 2 * self.min_samples_leaf:
                    node.is_leaf = True
                    self._diagnostics.nodes.append(NodeDiagnostics(
                        node_id=node.node_id, depth=node.depth, n_samples=n,
                        n_candidates=0, n_accepted=0, accept_rate=0.0,
                        rho_max=0.0, rho_mean=float("nan"), rho_median=float("nan"),
                        split_found=False,
                    ))
                else:
                    splittable.append((node, idx, g_n, h_n))

            if not splittable:
                continue

            # ---- Phase 2: single batched histogram + screening call ------
            batch_items = [(idx, g_n, h_n) for _, idx, g_n, h_n in splittable]
            results     = self._screen_batch(batch_items, X_binned, B)

            # ---- Phase 3: apply each split result -----------------------
            for (node, idx, g_n, h_n), result in zip(splittable, results):
                n  = len(idx)
                rho_flat = result.get("rho", np.array([result["best_rho"][0]])).ravel()
                accepted_vals = rho_flat[rho_flat > 0.0]
                n_cand = int((B - 1) * F)
                n_acc  = len(accepted_vals)
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

                best_feat   = int(result["best_feat"][0])
                best_bin    = int(result["best_bin"][0])
                default_dir = int(result["best_default_dir"][0])

                feat_vals  = X_binned[idx, best_feat]
                goes_right = feat_vals > best_bin
                if default_dir == 1:
                    goes_right = goes_right | (feat_vals < 0)
                left_mask  = ~goes_right
                right_mask = goes_right

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    node.is_leaf = True
                    continue

                node.split_feature   = best_feat
                node.split_bin       = best_bin
                node.split_threshold = self._binner.threshold(best_feat, best_bin)
                node.default_dir     = default_dir

                left_node  = _Node(next_id,     node.depth + 1, idx[left_mask])
                right_node = _Node(next_id + 1, node.depth + 1, idx[right_mask])
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
                x_val = X[i, node.split_feature]
                if np.isnan(x_val):
                    child_id = node.right_child if node.default_dir == 1 else node.left_child
                elif x_val <= node.split_threshold:
                    child_id = node.left_child
                else:
                    child_id = node.right_child
                node = node_map[child_id]
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
