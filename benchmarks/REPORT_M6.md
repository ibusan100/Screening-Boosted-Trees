# M6 Benchmark Report -- Triton GPU Acceleration

## Setup

| Item | Value |
|------|-------|
| Date | 2026-04-09 |
| Platform | Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Triton | triton-windows (same API as upstream) |
| sbt version | 0.1.0 |
| Reproduce | `python benchmarks/bench_triton_speed.py` |

## Kernel Microbenchmark (N=65536, F=8, B=255, 10% missing)

| Path | Time / call | Speedup |
|------|-------------|---------|
| NumPy (CPU) | 17.4 ms | 1x |
| **Triton (RTX 4060 Ti)** | **0.34 ms** | **51x** |

The 51x speedup covers the full histogram-build + missing-stats + screening
pipeline for a single node. At this scale (65k samples, 8 features, 255 bins)
the bottleneck shifts entirely from memory-bandwidth-limited NumPy scatter to
GPU compute.

## E2E ScreeningBooster -- California Housing (N=20640, oblivious, 100 rounds)

### M6 initial (per-node GPU calls)

| Backend | RMSE (mean +- std) | Train time | Speedup |
|---------|-------------------|------------|---------|
| CPU (NumPy) | 0.5286 +- 0.0039 | 4.95 s | 1x |
| CUDA (Triton, per-node) | 0.5286 +- 0.0039 | 3.68 s | 1.35x |

### M7 (batched multi-node GPU calls)

| Backend | RMSE (mean +- std) | Train time | Speedup |
|---------|-------------------|------------|---------|
| CPU (NumPy) | 0.5286 +- 0.0039 | 5.42 s | 1x |
| **CUDA (Triton, batched)** | **0.5286 +- 0.0039** | **2.77 s** | **1.96x** |

### M7+ (full GPU pipeline)

| Backend | RMSE (mean +- std) | Train time | Speedup |
|---------|-------------------|------------|---------|
| CPU (NumPy) | 0.5286 +- 0.0039 | 5.34 s | 1x |
| **CUDA (Triton, full GPU pipeline)** | **0.5286 +- 0.0039** | **1.70 s** | **3.15x** |

RMSE is identical to floating-point precision in all cases -- the Triton path is
numerically equivalent to the NumPy reference (CLAUDE.md Inv-2).

### M7 batching design

`_bfs_core` now drains the **entire BFS queue at once** per iteration instead of
one node at a time.  All splittable nodes in the frontier are packed into
contiguous arrays (one `np.concatenate` per array), sent to GPU in a single
transfer, and processed by one `build_histogram_triton` call with
`num_nodes = frontier_size`.  `ObliviousTree._find_level_split` similarly
replaces the per-node Python loop with a single batched call + `sum(dim=0)`.

Result: kernel launches per round drop from O(frontier_size) to O(1), and the
single host-to-device transfer shrinks from N (per-node) to N_total (all at once).

### M7+ full GPU pipeline design (2.14x → 3.15x)

Three further optimisations applied in one pass:

1. **X_binned cache (booster level)** — `ScreeningBooster.fit()` pre-computes
   `X_binned = binner.transform(X)` once and passes it to every round's
   `tree.fit_gradients(..., X_binned=X_binned)`, eliminating 100× `binner.transform`
   calls (previously O(N×F×log B) per round).

2. **X_gpu cache across rounds** — `X_gpu = torch.from_numpy(X_binned).cuda()` is
   created once in `ScreeningBooster.fit()` and reused every round via the
   `X_gpu=` kwarg, eliminating 100× GPU uploads of the 512 KB feature matrix.
   Each tree's `_maybe_upload_X()` now accepts an optional pre-loaded tensor.

3. **GPU gather + normalisation for g/h** — In CUDA mode, `_bfs_core` and
   `_find_level_split` skip the CPU `g_fn` call entirely.  Raw g/h tensors are
   pre-uploaded by the booster (`g_gpu=`, `h_gpu=`) then gathered and per-node
   normalised on-device via `normalize_gh_batched_gpu()` (scatter_add + sqrt).
   This eliminates all per-batch CPU normalisation and CPU→GPU g/h transfers.

### Remaining gap to theoretical maximum

The ~3x E2E speedup vs the ~51x kernel speedup is explained by:
1. **Python tree management** (sample routing, min_samples check) still runs on CPU
2. **Leaf value computation** (`g[idx].sum() / h[idx].sum()`) is CPU NumPy per node
3. **Result tensor slicing + extraction** (CPU after the batched kernel)
4. **idx/nid concatenation** (`np.concatenate` per batch) still on CPU

## Correctness tests (77 passed)

Triton consistency tests (19) cover:
- Kernel A (histogram): shape, values, MISSING_BIN skip
- Kernel A2 (missing scatter): shape, values, zero-when-no-missing
- Kernel B (screening): rho, best_feat/bin, accepted_mask, default_dir -- all
  tested with and without missing values, atol=1e-6 on same histogram input
- End-to-end A+A2->B: atol=1e-3 (Kernel A non-determinism dominates)
