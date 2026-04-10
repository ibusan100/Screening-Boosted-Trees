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
    build_missing_stats,
    screening_split_numpy,
)

_TRITON_AVAILABLE: bool | None = None


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
# Internal per-level split record
# ---------------------------------------------------------------------------


@dataclass
class _LevelSplit:
    depth: int
    feat: int
    bin_idx: int
    threshold: float
    default_dir: int = 0  # 0=missing→left, 1=missing→right


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
        device: str = "cpu",
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_bins = num_bins
        self.params = params if params is not None else ScreeningParams()
        self.screening_mode = screening_mode
        self._device = device

        self._binner: Optional[Binner] = None
        self._level_splits: list[_LevelSplit] = []
        self._leaf_values: Optional[np.ndarray] = None
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
    ) -> "ObliviousTree":
        """Standalone tree: gradients recentred + std-normalised per node."""
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
            g = (y_n - y_n.mean()).astype(np.float32)
            g_std = float(np.std(g)) + 1e-8
            return (g / g_std).astype(np.float32), np.ones(len(idx), dtype=np.float32)

        def leaf_fn(idx: np.ndarray) -> float:
            return float(np.mean(y[idx])) if len(idx) > 0 else 0.0

        self._maybe_upload_X(X_binned, X_gpu=X_gpu)
        self._fit_core(X_binned, g_fn, leaf_fn)
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
    ) -> "ObliviousTree":
        """Boosting mode: g/h supplied externally; leaf = −G/(H+λ).

        Parameters
        ----------
        X      : (N, F) float64
        g      : (N,)   float32 — first-order gradient from the loss.
        h      : (N,)   float32 — second-order gradient (hessian).
        binner : pre-fitted Binner to reuse across rounds.
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
            # CPU normalisation — only used in CPU mode (_find_level_split skips
            # this in CUDA mode and uses GPU gather+normalise instead).
            g_n = g[idx]
            g_c = g_n - g_n.mean()
            g_std = float(np.std(g_c)) + 1e-8
            h_n = h[idx]
            h_mean = float(h_n.mean()) + 1e-8
            return (g_c / g_std).astype(np.float32), (h_n / h_mean).astype(np.float32)

        def leaf_fn(idx: np.ndarray) -> float:
            if len(idx) == 0:
                return 0.0
            return float(-g[idx].sum() / (h[idx].sum() + lam))

        self._maybe_upload_X(X_binned, X_gpu=X_gpu)
        self._fit_core(X_binned, g_fn, leaf_fn)
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

            feat, bin_idx, default_dir, accepted = self._find_level_split(
                X_binned, g_fn, splittable, B, F, depth
            )

            if not accepted:
                break  # Screening rejected all level candidates

            threshold = self._binner.threshold(feat, bin_idx)
            self._level_splits.append(_LevelSplit(depth, feat, bin_idx, threshold, default_dir))

            # Split every node (including small ones — they're split passively)
            new_nodes: list[np.ndarray] = []
            for idx in nodes:
                feat_vals  = X_binned[idx, feat]
                goes_right = feat_vals > bin_idx        # NaN→-1, always False
                if default_dir == 1:                    # missing → right
                    goes_right = goes_right | (feat_vals < 0)
                new_nodes.append(idx[~goes_right])
                new_nodes.append(idx[goes_right])
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
    ) -> tuple[int, int, int, bool]:
        """Aggregate histograms across splittable nodes and apply screening once."""
        total_n = sum(len(idx) for idx in splittable)

        if self._device == "cuda":
            import torch
            from .kernels.screening_split_triton import (
                build_histogram_triton,
                build_missing_stats_triton,
                normalize_gh_batched_gpu,
                normalize_y_batched_gpu,
                screening_split_triton,
            )
            # Pack sample indices; use pre-loaded GPU tensors — no CPU X/g/h copies.
            K = len(splittable)
            all_idx = np.concatenate(splittable)
            all_nid = np.concatenate([
                np.full(len(idx), i, dtype=np.int32) for i, idx in enumerate(splittable)
            ])
            idx_t = torch.from_numpy(all_idx).long().cuda()
            nid_t = torch.from_numpy(all_nid).long().cuda()

            X_t = self._X_gpu[idx_t]              # GPU gather — no CPU X copy

            if self._mode == "boosting":
                g_raw = self._g_gpu[idx_t]         # GPU gather — no CPU g/h copy
                h_raw = self._h_gpu[idx_t]
                g_t, h_t = normalize_gh_batched_gpu(g_raw, h_raw, nid_t, K)
            else:  # standalone — centre + std-normalise on GPU
                y_raw = self._y_gpu[idx_t]
                g_t, h_t = normalize_y_batched_gpu(y_raw, nid_t, K, std_normalize=True)

            hG_all, hH_all = build_histogram_triton(X_t, g_t, h_t, nid_t.int(), K, num_bins)
            Gm_all, Hm_all = build_missing_stats_triton(X_t, g_t, h_t, nid_t.int(), K)

            # Aggregate across nodes (sum over node dimension → [1, F, B])
            agg_G  = hG_all.sum(dim=0, keepdim=True)
            agg_H  = hH_all.sum(dim=0, keepdim=True)
            agg_Gm = Gm_all.sum(dim=0, keepdim=True)
            agg_Hm = Hm_all.sum(dim=0, keepdim=True)

            out = screening_split_triton(agg_G, agg_H, self.params, agg_Gm, agg_Hm)
            result = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v)
                      for k, v in out.items()}
        else:
            agg_G  = np.zeros((1, F, num_bins), dtype=np.float32)
            agg_H  = np.zeros((1, F, num_bins), dtype=np.float32)
            agg_Gm = np.zeros((1, F), dtype=np.float32)
            agg_Hm = np.zeros((1, F), dtype=np.float32)
            for idx in splittable:
                g_norm, h_n = g_fn(idx)
                nid = np.zeros(len(idx), dtype=np.int32)
                X_node = X_binned[idx]
                hG, hH = build_histogram_numpy(X_node, g_norm, h_n, nid, 1, num_bins)
                Gm, Hm = build_missing_stats(X_node, g_norm, h_n, nid, num_nodes=1)
                agg_G += hG; agg_H += hH; agg_Gm += Gm; agg_Hm += Hm
            result = screening_split_numpy(agg_G, agg_H, self.params, agg_Gm, agg_Hm)

        # Diagnostics — one entry per level (node_id = depth for clarity)
        rho_flat = result.get("rho", np.array([result["best_rho"][0]])).ravel()
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
            return -1, -1, 0, False

        best_feat   = int(result["best_feat"][0])
        best_bin    = int(result["best_bin"][0])
        default_dir = int(result["best_default_dir"][0])
        return best_feat, best_bin, default_dir, True

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
            goes_right = X[:, split.feat] > split.threshold  # NaN → False (left)
            if split.default_dir == 1:                        # missing → right
                goes_right = goes_right | np.isnan(X[:, split.feat])
            leaf_idx = leaf_idx * 2 + goes_right.astype(np.int32)

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
