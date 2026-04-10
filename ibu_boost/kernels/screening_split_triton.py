"""
Triton fused kernels for Screening Boosted Trees — M6.

Three kernels:

  A. _hist_scatter_kernel
        Sample-parallel atomic scatter into a [num_nodes, F, num_bins]
        gradient/hessian histogram.  Samples with bin < 0 (MISSING_BIN)
        are skipped — their stats are accumulated separately by Kernel A2.

  A2. _missing_scatter_kernel
        Same sample-parallel scatter, but accumulates ONLY missing-bin
        samples into a [num_nodes, F] G_miss / H_miss array.

  B. _screening_split_kernel
        Per (node, feature) grid.  Each program loads one histogram row
        (B bins), computes the cumsum scan, evaluates the XGBoost-style
        missing-direction gain (both miss→left and miss→right), applies
        the bounded-gain + Trim-and-Square screening transform, and writes
        per-(node, feat) winners including the default direction.

A final per-node reduction over features is done on the host (cheap, F
values per node).

Invariants (CLAUDE.md § Inv-2):
  Kernel A  atol 1e-3  (non-deterministic atomic_add)
  Kernel B  atol 1e-6  (deterministic given same histogram)
  End-to-end atol 1e-3 (Kernel A dominates)

Windows: install with `pip install -e ".[triton]"` — pyproject.toml
selects triton-windows on win32 automatically.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    return 1 if x <= 1 else 1 << math.ceil(math.log2(x))

from ..screening_split import ScreeningParams


# ---------------------------------------------------------------------------
# Kernel A: histogram scatter (MISSING_BIN aware)
# ---------------------------------------------------------------------------


@triton.jit
def _hist_scatter_kernel(
    X_ptr,            # [N, F] int32 (binned; -1 = MISSING_BIN)
    G_ptr,            # [N] f32
    H_ptr,            # [N] f32
    NID_ptr,          # [N] int32
    HG_ptr,           # [num_nodes, F, B] f32 — output
    HH_ptr,
    N,
    F: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    g   = tl.load(G_ptr   + offs, mask=mask, other=0.0)
    h   = tl.load(H_ptr   + offs, mask=mask, other=0.0)
    nid = tl.load(NID_ptr + offs, mask=mask, other=0)

    for f in tl.static_range(F):
        x = tl.load(X_ptr + offs * F + f, mask=mask, other=-1)
        # Skip missing-bin samples (x < 0)
        valid = mask & (x >= 0)
        slot = (nid * F + f) * B + x
        tl.atomic_add(HG_ptr + slot, g, mask=valid)
        tl.atomic_add(HH_ptr + slot, h, mask=valid)


# ---------------------------------------------------------------------------
# Kernel A2: missing-value scatter
# ---------------------------------------------------------------------------


@triton.jit
def _missing_scatter_kernel(
    X_ptr,            # [N, F] int32
    G_ptr,            # [N] f32
    H_ptr,            # [N] f32
    NID_ptr,          # [N] int32
    GM_ptr,           # [num_nodes, F] f32 — output G_miss
    HM_ptr,           # [num_nodes, F] f32 — output H_miss
    N,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    g   = tl.load(G_ptr   + offs, mask=mask, other=0.0)
    h   = tl.load(H_ptr   + offs, mask=mask, other=0.0)
    nid = tl.load(NID_ptr + offs, mask=mask, other=0)

    for f in tl.static_range(F):
        x = tl.load(X_ptr + offs * F + f, mask=mask, other=0)
        miss_mask = mask & (x < 0)
        slot = nid * F + f
        tl.atomic_add(GM_ptr + slot, g, mask=miss_mask)
        tl.atomic_add(HM_ptr + slot, h, mask=miss_mask)


# ---------------------------------------------------------------------------
# Kernel B: cumsum + both-direction gain + screening + per-(node,feat) argmax
# ---------------------------------------------------------------------------


@triton.jit
def _screening_split_kernel(
    HG_ptr,               # [num_nodes, F, B_ACT] f32
    HH_ptr,
    GM_ptr,               # [num_nodes, F] f32 — G_miss per (node, feat)
    HM_ptr,               # [num_nodes, F] f32 — H_miss per (node, feat)
    BEST_RHO_ptr,         # [num_nodes, F] f32 — output
    BEST_BIN_ptr,         # [num_nodes, F] i32 — output
    BEST_DIR_ptr,         # [num_nodes, F] i32 — output (0=miss->left, 1=miss->right)
    s_w: tl.constexpr,
    s_r: tl.constexpr,
    lam,
    eps,
    F:      tl.constexpr,
    B_ACT:  tl.constexpr,   # actual number of bins (may not be a power of 2)
    B_POW2: tl.constexpr,   # next power of 2 >= B_ACT, used for tl.arange / tl.cumsum
):
    nid = tl.program_id(0)
    fid = tl.program_id(1)

    # tl.arange requires a power-of-2 range; pad to B_POW2 and mask extras.
    bins  = tl.arange(0, B_POW2)
    valid = bins < B_ACT      # mask for bins that actually exist
    base  = (nid * F + fid) * B_ACT

    g = tl.load(HG_ptr + base + bins, mask=valid, other=0.0)
    h = tl.load(HH_ptr + base + bins, mask=valid, other=0.0)

    G_total = tl.sum(g, axis=0)
    H_total = tl.sum(h, axis=0)

    # Missing stats for this (node, feat)
    mf_idx = nid * F + fid
    gm = tl.load(GM_ptr + mf_idx)
    hm = tl.load(HM_ptr + mf_idx)

    parent = (G_total * G_total) / (H_total + lam)

    G_L = tl.cumsum(g, axis=0)
    H_L = tl.cumsum(h, axis=0)
    G_R = G_total - G_L
    H_R = H_total - H_L

    # XGBoost-style: evaluate both missing-value directions, take the better one.
    gain_A = (G_L + gm) * (G_L + gm) / (H_L + hm + lam) + G_R * G_R / (H_R + lam) - parent
    gain_B = G_L * G_L / (H_L + lam) + (G_R + gm) * (G_R + gm) / (H_R + hm + lam) - parent
    raw = tl.maximum(gain_A, gain_B)
    dir_right = gain_B > gain_A  # True → miss goes right at this split

    # Normalise by node size → N-invariant, O(1) scale
    norm = raw / tl.maximum(H_total, 1.0)
    norm = tl.maximum(norm, 0.0)

    tau = tl.exp(s_w) + eps
    r   = tl.exp(s_r) + 1.0
    s   = 1.0 - tl.exp(-norm / tau)
    rho = tl.maximum(1.0 - r * (1.0 - s), 0.0)
    rho = rho * rho

    # Zero out: last actual bin (no right child) + padding bins
    rho = tl.where(valid & (bins < (B_ACT - 1)), rho, 0.0)

    best_idx = tl.argmax(rho, axis=0)
    best_val = tl.max(rho, axis=0)

    # Gather default_dir at the winning bin.
    best_dir = tl.sum(tl.where(bins == best_idx, dir_right.to(tl.int32), 0), axis=0)

    out_idx = nid * F + fid
    tl.store(BEST_RHO_ptr + out_idx, best_val)
    tl.store(BEST_BIN_ptr + out_idx, best_idx.to(tl.int32))
    tl.store(BEST_DIR_ptr + out_idx, best_dir)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def build_histogram_triton(
    X_binned: torch.Tensor,   # [N, F] int32 on CUDA  (MISSING_BIN=-1 for NaN)
    g: torch.Tensor,          # [N] f32
    h: torch.Tensor,          # [N] f32
    node_id: torch.Tensor,    # [N] int32
    num_nodes: int,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter g/h into per-(node, feat, bin) histograms; MISSING_BIN samples skipped."""
    assert X_binned.is_cuda, "Triton path requires CUDA tensors"
    N, F = X_binned.shape
    X_binned = X_binned.contiguous()
    g        = g.contiguous()
    h        = h.contiguous()
    node_id  = node_id.contiguous()

    hist_G = torch.zeros((num_nodes, F, num_bins), device=X_binned.device, dtype=torch.float32)
    hist_H = torch.zeros_like(hist_G)

    BLOCK = 1024
    grid  = (triton.cdiv(N, BLOCK),)
    _hist_scatter_kernel[grid](
        X_binned, g, h, node_id, hist_G, hist_H,
        N, F=F, B=num_bins, BLOCK=BLOCK,
    )
    return hist_G, hist_H


def build_missing_stats_triton(
    X_binned: torch.Tensor,   # [N, F] int32 on CUDA
    g: torch.Tensor,          # [N] f32
    h: torch.Tensor,          # [N] f32
    node_id: torch.Tensor,    # [N] int32
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate g/h for MISSING_BIN samples into per-(node, feat) arrays.

    Returns
    -------
    G_miss : [num_nodes, F] float32
    H_miss : [num_nodes, F] float32
    """
    assert X_binned.is_cuda, "Triton path requires CUDA tensors"
    N, F = X_binned.shape

    G_miss = torch.zeros((num_nodes, F), device=X_binned.device, dtype=torch.float32)
    H_miss = torch.zeros_like(G_miss)

    BLOCK = 1024
    grid  = (triton.cdiv(N, BLOCK),)
    _missing_scatter_kernel[grid](
        X_binned.contiguous(), g.contiguous(), h.contiguous(), node_id.contiguous(),
        G_miss, H_miss,
        N, F=F, BLOCK=BLOCK,
    )
    return G_miss, H_miss


def screening_split_triton(
    hist_G: torch.Tensor,                     # [num_nodes, F, B]
    hist_H: torch.Tensor,
    params: ScreeningParams,
    G_miss: Optional[torch.Tensor] = None,    # [num_nodes, F]
    H_miss: Optional[torch.Tensor] = None,
) -> dict:
    """Run the screening transform on pre-built histograms.

    Returns a dict compatible with screening_split_numpy:
        best_feat          [num_nodes] int32
        best_bin           [num_nodes] int32
        best_rho           [num_nodes] float32
        accepted_mask      [num_nodes] bool
        best_default_dir   [num_nodes] int8  (0=miss→left, 1=miss→right)
    """
    assert hist_G.is_cuda
    num_nodes, F, B = hist_G.shape

    dev = hist_G.device
    if G_miss is None:
        G_miss = torch.zeros((num_nodes, F), device=dev, dtype=torch.float32)
        H_miss = torch.zeros_like(G_miss)

    best_rho_per_feat = torch.empty((num_nodes, F), device=dev, dtype=torch.float32)
    best_bin_per_feat = torch.empty((num_nodes, F), device=dev, dtype=torch.int32)
    best_dir_per_feat = torch.empty((num_nodes, F), device=dev, dtype=torch.int32)

    B_POW2 = _next_pow2(B)
    grid   = (num_nodes, F)
    _screening_split_kernel[grid](
        hist_G.contiguous(), hist_H.contiguous(),
        G_miss.contiguous(), H_miss.contiguous(),
        best_rho_per_feat, best_bin_per_feat, best_dir_per_feat,
        float(params.s_w), float(params.s_r),
        float(params.lam), float(params.eps),
        F=F, B_ACT=B, B_POW2=B_POW2,
    )

    # Host-side reduction: argmax over feature axis (F values per node — trivial)
    best_rho, best_feat = best_rho_per_feat.max(dim=1)
    best_bin = best_bin_per_feat.gather(1, best_feat.unsqueeze(1)).squeeze(1)
    best_dir = best_dir_per_feat.gather(1, best_feat.unsqueeze(1)).squeeze(1)
    accepted = best_rho > 0.0

    return {
        "best_feat":        best_feat.to(torch.int32),
        "best_bin":         best_bin.to(torch.int32),
        "best_rho":         best_rho,
        "accepted_mask":    accepted,
        "best_default_dir": best_dir.to(torch.int8),
    }


# ---------------------------------------------------------------------------
# Host helpers: batched per-node GPU normalisation
# ---------------------------------------------------------------------------


def normalize_gh_batched_gpu(
    g_raw: torch.Tensor,   # [N_total] float32 — raw gradients (gathered)
    h_raw: torch.Tensor,   # [N_total] float32 — raw hessians (gathered)
    nid_t: torch.Tensor,   # [N_total] int64  — node assignment per sample
    K: int,                # number of nodes in this batch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boosting mode: per-node normalise g (mean=0, std=1) and h (mean=1) on GPU.

    Equivalent to ScreeningTree.fit_gradients' g_fn but fully on-device;
    no CPU→GPU transfer of g/h is needed when the raw tensors are pre-loaded.
    """
    nid = nid_t.long()
    dev = g_raw.device
    counts = torch.bincount(nid, minlength=K).float()        # [K]

    # g: centre per node
    g_sum = torch.zeros(K, device=dev, dtype=torch.float32).scatter_add_(0, nid, g_raw)
    g_mean = g_sum / counts                                   # [K]
    g_c = g_raw - g_mean[nid]

    # g: std per node
    g_sq = torch.zeros(K, device=dev, dtype=torch.float32).scatter_add_(0, nid, g_c * g_c)
    g_std = (g_sq / counts).sqrt() + 1e-8                    # [K]
    g_norm = g_c / g_std[nid]

    # h: mean per node
    h_sum = torch.zeros(K, device=dev, dtype=torch.float32).scatter_add_(0, nid, h_raw)
    h_mean = h_sum / counts + 1e-8                           # [K]
    h_norm = h_raw / h_mean[nid]

    return g_norm, h_norm


def normalize_y_batched_gpu(
    y_raw: torch.Tensor,      # [N_total] float32 — raw targets (gathered)
    nid_t: torch.Tensor,      # [N_total] int64
    K: int,
    std_normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone mode: per-node centre y (subtract mean) → g; h = ones.

    std_normalize=True: also divide g by per-node std
    (ObliviousTree standalone uses std=1 normalisation; ScreeningTree does not).
    """
    nid = nid_t.long()
    dev = y_raw.device
    counts = torch.bincount(nid, minlength=K).float()

    y_sum = torch.zeros(K, device=dev, dtype=torch.float32).scatter_add_(0, nid, y_raw)
    y_mean = y_sum / counts                                   # [K]
    g = y_raw - y_mean[nid]

    if std_normalize:
        g_sq = torch.zeros(K, device=dev, dtype=torch.float32).scatter_add_(0, nid, g * g)
        g_std = (g_sq / counts).sqrt() + 1e-8
        g = g / g_std[nid]

    h = torch.ones_like(g)
    return g, h
