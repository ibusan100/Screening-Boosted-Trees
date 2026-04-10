"""
Screening Boosted Trees — split scoring (NumPy reference).

The screening transform mirrors "Screening Is Enough" (Nakanishi 2026,
arXiv:2604.01178) but applied to GBDT split selection rather than attention:

  raw_gain  = G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G_total^2/(H_total+lam)
  norm_gain = raw_gain / max(H_total, 1)       # normalise by node size → O(1), N-invariant
  s         = 1 - exp(-norm_gain / tau)        # bounded "similarity" in [0, 1)
  rho       = max(1 - r * (1 - s), 0) ** 2    # Trim-and-Square (absolute screening)

  tau = exp(s_w) + eps           (learned scalar; analogue of the screening window)
  r   = exp(s_r) + 1             (learned scalar; 1/r is the acceptance width)

Normalising by H_total (= N_node for MSE where h_i=1) makes norm_gain O(1)
regardless of dataset size, so defaults transfer across tasks.  Analogue:
QK unit-normalisation in Multiscreen bounds similarity to [-1,1] so a fixed
threshold is meaningful.

A node selects argmax_{(feature, bin)} rho. If max(rho) == 0 the node is
**rejected** — no split is emitted, the node becomes a leaf. This is the GBDT
analogue of "no key is relevant" in Multiscreen: weak splits are removed
cleanly without an external min_gain_to_split heuristic, and the threshold is
itself learnable through s_r.

This module contains the pure-NumPy reference. The Triton-accelerated path
lives in `sbt.kernels.screening_split_triton` and is loaded lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScreeningParams:
    """Learned (or fixed) screening scalars and L2 regularizer.

    s_w, s_r are stored in log space exactly as in the Multiscreen paper:
        tau = exp(s_w) + eps
        r   = exp(s_r) + 1
    """

    # Defaults calibrated after H_total normalisation (gain is O(1), N-invariant):
    #   s_w=-2.0, s_r=0.0 → tau≈0.135, r=2.0 → root accept_rate ≈ 15%
    #   Stable across N=300..20640 (tested). Raise s_w to reject more aggressively;
    #   lower s_r (closer to 0) to accept more.
    s_w: float = -2.0     # log-scale gain temperature; tau = exp(s_w) + eps
    s_r: float = 0.0      # log-scale acceptance width; r = exp(s_r) + 1
    lam: float = 1.0      # L2 reg on hessian
    eps: float = 1e-6

    def tau(self) -> float:
        return float(np.exp(self.s_w)) + self.eps

    def r(self) -> float:
        return float(np.exp(self.s_r)) + 1.0


_MISSING_BIN: int = -1  # mirrors binning.MISSING_BIN; kept local to avoid circular import


def build_histogram_numpy(
    X_binned: np.ndarray,   # [N, F] int — pre-binned features (MISSING_BIN=-1 for NaN)
    g: np.ndarray,          # [N] float32 — gradients
    h: np.ndarray,          # [N] float32 — hessians
    node_id: np.ndarray,    # [N] int32 — leaf membership
    num_nodes: int,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-(node, feature, bin) gradient / hessian histograms.

    Samples with X_binned[:, f] == MISSING_BIN (-1) are skipped.
    Call build_missing_stats() separately to obtain their aggregates.
    """
    N, F = X_binned.shape
    hist_G = np.zeros((num_nodes, F, num_bins), dtype=np.float32)
    hist_H = np.zeros((num_nodes, F, num_bins), dtype=np.float32)
    for f in range(F):
        valid = X_binned[:, f] >= 0
        np.add.at(hist_G, (node_id[valid], f, X_binned[valid, f]), g[valid])
        np.add.at(hist_H, (node_id[valid], f, X_binned[valid, f]), h[valid])
    return hist_G, hist_H


def build_missing_stats(
    X_binned: np.ndarray,   # [N, F] int — MISSING_BIN=-1 marks NaN entries
    g: np.ndarray,          # [N] float32
    h: np.ndarray,          # [N] float32
    node_id: np.ndarray,    # [N] int32
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum g and h for missing-value samples per (node, feature).

    Returns
    -------
    G_miss : [num_nodes, F] float32
    H_miss : [num_nodes, F] float32
    """
    N, F = X_binned.shape
    G_miss = np.zeros((num_nodes, F), dtype=np.float32)
    H_miss = np.zeros((num_nodes, F), dtype=np.float32)
    for f in range(F):
        miss = X_binned[:, f] < 0
        if miss.any():
            np.add.at(G_miss[:, f], node_id[miss], g[miss])
            np.add.at(H_miss[:, f], node_id[miss], h[miss])
    return G_miss, H_miss


def _bounded_gain(raw_gain: np.ndarray, tau: float) -> np.ndarray:
    raw_gain = np.maximum(raw_gain, 0.0)
    return 1.0 - np.exp(-raw_gain / tau)


def _trim_square(s: np.ndarray, r: float) -> np.ndarray:
    return np.square(np.maximum(1.0 - r * (1.0 - s), 0.0))


def screening_split_numpy(
    hist_G: np.ndarray,                     # [num_nodes, F, num_bins]
    hist_H: np.ndarray,
    params: ScreeningParams,
    G_miss: Optional[np.ndarray] = None,    # [num_nodes, F] — optional missing stats
    H_miss: Optional[np.ndarray] = None,
) -> dict:
    """Reference implementation: cumsum scan + Trim-and-Square per (node, feat),
    then per-node reduction across (feat, bin).

    When G_miss / H_miss are provided, each candidate split evaluates BOTH
    possible directions for missing values (left or right) and uses the
    direction that gives the higher gain (XGBoost-style default direction).

    Returns
    -------
    dict with:
        rho            [num_nodes, F, num_bins]   screened relevance
        best_feat      [num_nodes]
        best_bin       [num_nodes]
        best_rho       [num_nodes]
        accepted_mask  [num_nodes]
        default_dir    [num_nodes, F, num_bins]   0=miss→left, 1=miss→right
                                                  (only meaningful when G_miss given)
        grad_sw        [num_nodes, F, num_bins]   ∂rho/∂s_w (closed-form)
        grad_sr        [num_nodes, F, num_bins]   ∂rho/∂s_r
    """
    from typing import Optional as _Opt  # noqa: F401 (already imported at top)

    num_nodes, F, B = hist_G.shape
    tau = params.tau()
    r = params.r()
    lam = params.lam

    G_total = hist_G.sum(axis=2, keepdims=True)   # [N, F, 1]
    H_total = hist_H.sum(axis=2, keepdims=True)

    G_L = np.cumsum(hist_G, axis=2)
    H_L = np.cumsum(hist_H, axis=2)
    G_R = G_total - G_L
    H_R = H_total - H_L

    if G_miss is not None and H_miss is not None:
        # Expand to [num_nodes, F, 1] for broadcasting
        Gm = G_miss[:, :, np.newaxis]
        Hm = H_miss[:, :, np.newaxis]
        parent = (G_total ** 2) / (H_total + lam)  # G_total unchanged by missing routing

        # Option A: missing → left
        gain_A = (
            (G_L + Gm) ** 2 / (H_L + Hm + lam)
            + G_R ** 2 / (H_R + lam)
            - parent
        )
        # Option B: missing → right
        gain_B = (
            G_L ** 2 / (H_L + lam)
            + (G_R + Gm) ** 2 / (H_R + Hm + lam)
            - parent
        )
        raw_gain = np.maximum(gain_A, gain_B)
        default_dir = (gain_B > gain_A).astype(np.int8)  # 0=left, 1=right
    else:
        parent = (G_total ** 2) / (H_total + lam)
        raw_gain = (G_L ** 2) / (H_L + lam) + (G_R ** 2) / (H_R + lam) - parent
        default_dir = np.zeros((num_nodes, F, B), dtype=np.int8)

    raw_gain[..., -1] = -np.inf   # last bin: no right child
    default_dir[..., -1] = 0

    norm_gain = raw_gain / np.maximum(H_total, 1.0)

    s = _bounded_gain(norm_gain, tau)
    rho = _trim_square(s, r)
    rho[..., -1] = 0.0
    default_dir[..., -1] = 0

    # ------------------------------------------------------------------ #
    # Closed-form gradients ∂rho/∂s_w and ∂rho/∂s_r (Phase 3 groundwork)  #
    # ∂rho/∂s_w = 2*sqrt(rho)*r * (ng*(1-s)/tau) * exp(s_w)               #
    #           = 2*sqrt(rho)*r * ng*(1-s)/tau * (tau-eps)                 #
    # ∂rho/∂s_r = -2*sqrt(rho)*(1-s) * (r-1)                              #
    # Both are 0 where rho == 0 (non-differentiable, subgradient=0).       #
    # ------------------------------------------------------------------ #
    eps = params.eps
    sqrt_rho = np.sqrt(np.maximum(rho, 0.0))
    ng_clipped = np.maximum(norm_gain, 0.0)

    grad_sw = np.where(
        rho > 0.0,
        2.0 * sqrt_rho * r * (ng_clipped * (1.0 - s) / tau) * (tau - eps),
        0.0,
    ).astype(np.float32)

    grad_sr = np.where(
        rho > 0.0,
        -2.0 * sqrt_rho * (1.0 - s) * (r - 1.0),
        0.0,
    ).astype(np.float32)

    flat = rho.reshape(num_nodes, F * B)
    best_flat = flat.argmax(axis=1)
    best_feat = (best_flat // B).astype(np.int32)
    best_bin = (best_flat % B).astype(np.int32)
    best_rho = flat.max(axis=1).astype(np.float32)
    accepted = best_rho > 0.0

    # Per-node scalar: default direction at the winning (feat, bin).
    # Exposed as "best_default_dir" so callers don't need to index into the
    # full [num_nodes, F, B] default_dir array — this key is also returned by
    # the Triton path, enabling a single code path in ScreeningTree / ObliviousTree.
    best_default_dir = np.array(
        [int(default_dir[n, best_feat[n], best_bin[n]]) for n in range(num_nodes)],
        dtype=np.int8,
    )

    return {
        "rho": rho,
        "best_feat": best_feat,
        "best_bin": best_bin,
        "best_rho": best_rho,
        "accepted_mask": accepted,
        "default_dir": default_dir,
        "best_default_dir": best_default_dir,
        "grad_sw": grad_sw,
        "grad_sr": grad_sr,
    }
