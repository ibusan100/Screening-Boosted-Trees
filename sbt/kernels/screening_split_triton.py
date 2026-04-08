"""
Triton fused kernels for Screening Boosted Trees.

Two kernels:

  A. _hist_scatter_kernel
        Sample-parallel atomic scatter into a [num_nodes, F, num_bins]
        gradient/hessian histogram. Each program handles a BLOCK of samples
        and unrolls over F (small).

  B. _screening_split_kernel
        Per (node, feature) cumsum scan over bins, computes raw gain,
        applies the bounded-gain + Trim-and-Square screening transform from
        the Multiscreen paper, and writes per-(node, feat) winners.

A final per-node reduction over features is done on the host (cheap, just
F values per node).

Windows users: install with `pip install screening-boosted-trees[triton]`,
which selects `triton-windows` automatically via the platform marker in
pyproject.toml. The kernel source is identical — `triton-windows` ships the
same `triton` / `triton.language` API as upstream Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..screening_split import ScreeningParams


# ---------------------------------------------------------------------------
# Kernel A: histogram scatter
# ---------------------------------------------------------------------------


@triton.jit
def _hist_scatter_kernel(
    X_ptr,            # [N, F] int32 (binned)
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

    g = tl.load(G_ptr + offs, mask=mask, other=0.0)
    h = tl.load(H_ptr + offs, mask=mask, other=0.0)
    nid = tl.load(NID_ptr + offs, mask=mask, other=0)

    # F is small (typically tens). Unroll the per-feature scatter.
    for f in tl.static_range(F):
        x = tl.load(X_ptr + offs * F + f, mask=mask, other=0)
        slot = (nid * F + f) * B + x
        tl.atomic_add(HG_ptr + slot, g, mask=mask)
        tl.atomic_add(HH_ptr + slot, h, mask=mask)


# ---------------------------------------------------------------------------
# Kernel B: cumsum + bounded gain + Trim-and-Square + per-(node,feat) argmax
# ---------------------------------------------------------------------------


@triton.jit
def _screening_split_kernel(
    HG_ptr,           # [num_nodes, F, B] f32
    HH_ptr,
    BEST_RHO_ptr,     # [num_nodes, F] f32
    BEST_BIN_ptr,     # [num_nodes, F] i32
    s_w,              # learned scalar (log-space gain temperature)
    s_r,              # learned scalar (log-space acceptance width)
    lam,              # L2 reg (float)
    eps,              # float
    F: tl.constexpr,
    B: tl.constexpr,
):
    nid = tl.program_id(0)
    fid = tl.program_id(1)

    bins = tl.arange(0, B)
    base = (nid * F + fid) * B
    g = tl.load(HG_ptr + base + bins)
    h = tl.load(HH_ptr + base + bins)

    G_total = tl.sum(g, axis=0)
    H_total = tl.sum(h, axis=0)
    parent = (G_total * G_total) / (H_total + lam)

    G_L = tl.cumsum(g, axis=0)
    H_L = tl.cumsum(h, axis=0)
    G_R = G_total - G_L
    H_R = H_total - H_L

    raw = (G_L * G_L) / (H_L + lam) + (G_R * G_R) / (H_R + lam) - parent
    # Normalise by node size (H_total) → N-invariant, O(1) scale.
    norm = raw / tl.maximum(H_total, 1.0)
    norm = tl.maximum(norm, 0.0)

    tau = tl.exp(s_w) + eps
    r = tl.exp(s_r) + 1.0
    s = 1.0 - tl.exp(-norm / tau)
    rho = tl.maximum(1.0 - r * (1.0 - s), 0.0)
    rho = rho * rho

    # The last bin has no right child — exclude it from the argmax.
    last_mask = bins < (B - 1)
    rho = tl.where(last_mask, rho, 0.0)

    best_idx = tl.argmax(rho, axis=0)
    best_val = tl.max(rho, axis=0)

    tl.store(BEST_RHO_ptr + nid * F + fid, best_val)
    tl.store(BEST_BIN_ptr + nid * F + fid, best_idx.to(tl.int32))


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def build_histogram_triton(
    X_binned: torch.Tensor,   # [N, F] int32 on CUDA
    g: torch.Tensor,          # [N] f32
    h: torch.Tensor,          # [N] f32
    node_id: torch.Tensor,    # [N] int32
    num_nodes: int,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert X_binned.is_cuda, "Triton path requires CUDA tensors"
    N, F = X_binned.shape
    X_binned = X_binned.contiguous()
    g = g.contiguous()
    h = h.contiguous()
    node_id = node_id.contiguous()

    hist_G = torch.zeros((num_nodes, F, num_bins), device=X_binned.device, dtype=torch.float32)
    hist_H = torch.zeros_like(hist_G)

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _hist_scatter_kernel[grid](
        X_binned, g, h, node_id, hist_G, hist_H,
        N, F=F, B=num_bins, BLOCK=BLOCK,
    )
    return hist_G, hist_H


def screening_split_triton(
    hist_G: torch.Tensor,     # [num_nodes, F, B]
    hist_H: torch.Tensor,
    params: ScreeningParams,
) -> dict:
    assert hist_G.is_cuda
    num_nodes, F, B = hist_G.shape

    best_rho_per_feat = torch.empty((num_nodes, F), device=hist_G.device, dtype=torch.float32)
    best_bin_per_feat = torch.empty((num_nodes, F), device=hist_G.device, dtype=torch.int32)

    grid = (num_nodes, F)
    _screening_split_kernel[grid](
        hist_G, hist_H, best_rho_per_feat, best_bin_per_feat,
        float(params.s_w), float(params.s_r), float(params.lam), float(params.eps),
        F=F, B=B,
    )

    # Final reduction across features (host side; F values per node — trivial).
    best_rho, best_feat = best_rho_per_feat.max(dim=1)
    best_bin = best_bin_per_feat.gather(1, best_feat.unsqueeze(1)).squeeze(1)
    accepted = best_rho > 0.0

    return {
        "best_feat": best_feat.to(torch.int32),
        "best_bin": best_bin.to(torch.int32),
        "best_rho": best_rho,
        "accepted_mask": accepted,
    }
