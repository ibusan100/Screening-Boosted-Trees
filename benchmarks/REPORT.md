# M2 Benchmark Report — ScreeningBooster vs LightGBM vs XGBoost

## Setup

| Item | Value |
|------|-------|
| Date | 2026-04-08 |
| Platform | Windows 11 Home 10.0.26200 |
| Python | 3.11 |
| sbt version | 0.0.1 (NumPy reference, no Triton) |
| Dataset | California housing (`sklearn.datasets.fetch_california_housing`) |
| N / F | 20 640 samples, 8 features, target = median house value |
| Split | 80% train (16 512) / 20% test (4 128), `random_state ∈ {0, 1, 2}` |
| Reproduce | `python benchmarks/bench_vs_baseline.py` |

## Hyperparameters (matched across all models)

| Param | Value |
|-------|-------|
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| lambda (L2 reg) | 1.0 |
| min_samples_leaf | 20 |
| num_bins | 255 |
| SBT s_w / s_r | −2.0 / −6.0 |

## Results

| Model | RMSE (mean ± std) | Train time (mean ± std) | Notes |
|-------|-------------------|-------------------------|-------|
| **ScreeningBooster** | **0.4881 ± 0.0048** | 3.0 ± 0.1 s | NumPy only |
| LightGBM 4.x | 0.4711 ± 0.0042 | 0.5 ± 0.6 s | C++ GBDT |
| XGBoost 2.x | 0.4713 ± 0.0047 | 0.10 ± 0.05 s | C++ GBDT |

SBT RMSE is within **3.6%** of LightGBM/XGBoost — competitive for a pure NumPy
reference with no hand-tuned split-selection heuristics.  The gap is expected to
narrow with Triton acceleration (M3) and eventual s_w/s_r learning (Phase 2/3).

## Screening diagnostics

Per-node gradient normalization (mean=0, std=1 within each node) makes
`norm_gain ∈ [0, 1]` independent of boosting round and dataset scale.

With `s_w=-2.0, s_r=-6.0` on seed=0:

| Metric | Value |
|--------|-------|
| Active rounds (n_leaves > 1) | 79 / 100 |
| Root accept_rate round 0 | ≈ 80% |
| Root accept_rate round 79 | ≈ 0% (auto-stop) |
| Mean leaves per active round | ≈ 40 |
| Mean accept_rate (all rounds) | 35.7% |

**Auto-stop property**: the screening transform naturally drives root
`accept_rate → 0` as residual correlations with features fall below threshold.
The last ≈ 20 rounds produce single-leaf trees (no-op updates) without any
external early-stopping logic.

## Parameter calibration note

`s_r=0.0` (standalone-tree default, `r=2`, `threshold_s=0.5`) rejects all
splits in later boosting rounds because per-round gains — even after gradient
normalization — decay as the model converges.  `s_r=-6.0` (`r≈1.003`,
`threshold_s≈0.0025`) preserves the auto-stop property while allowing the
~79 active rounds needed for competitive RMSE.

The two-parameter relationship:

```
threshold on norm_gain = tau × ln(r) = exp(s_w) × ln(exp(s_r) + 1)
  s_w=-2, s_r=0.0  →  0.135 × ln(2)     ≈ 0.094  (standalone trees)
  s_w=-2, s_r=-6.0 →  0.135 × ln(1.003) ≈ 0.0004 (boosting, 100 rounds)
```
