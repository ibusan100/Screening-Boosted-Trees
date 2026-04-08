# M3 Benchmark Report — ObliviousTree + Binary Classification

## Setup

| Item | Value |
|------|-------|
| Date | 2026-04-08 |
| Platform | Windows 11 Home 10.0.26200 |
| Python | 3.11 |
| sbt version | 0.0.1 (NumPy reference, no Triton) |
| Reproduce | `python benchmarks/bench_classification.py` |

## Hyperparameters (matched across all models)

| Param | Value |
|-------|-------|
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| lambda (L2 reg) | 1.0 |
| min_samples_leaf | 20 |
| SBT s_w / s_r | −2.0 / −6.0 |

## Datasets

| Dataset | N | F | Task | Source |
|---------|---|---|------|--------|
| Titanic | 1 309 | 6 | Binary classification | OpenML id=40945 |
| Adult (Census Income) | 48 842 | 14 | Binary classification | OpenML id=1590 |

Features: ordinal-encoded categoricals, median-imputed missing values.
Target: Titanic survival (0/1), Adult income >50K (0/1).

## Results — Titanic

| Model | AUC-ROC (mean ± std) | Accuracy | Train time |
|-------|----------------------|----------|------------|
| **SBT non-oblivious** | 0.7578 ± 0.0868 | 0.733 | 0.16 s |
| **SBT oblivious** | 0.7599 ± 0.0893 | 0.724 | 0.23 s |
| LightGBM | 0.8689 ± 0.0183 | 0.824 | 0.43 s |
| XGBoost | 0.8704 ± 0.0183 | 0.810 | 0.08 s |

Note: Titanic has only ~1 300 samples → high variance across seeds. 3-seed
std of 0.087 makes direct comparison inconclusive. The dataset is included
for interest (as requested), not as the primary evaluation.

## Results — Adult (primary evaluation)

| Model | AUC-ROC (mean ± std) | Accuracy | Train time |
|-------|----------------------|----------|------------|
| **SBT non-oblivious** | 0.7209 ± 0.0232 | 0.721 | 10.7 s |
| **SBT oblivious** | 0.8285 ± 0.0199 | 0.798 | 9.5 s |
| LightGBM | 0.9282 ± 0.0006 | 0.873 | 0.19 s |
| XGBoost | 0.9250 ± 0.0006 | 0.871 | 0.12 s |

## Key Findings

### 1. Oblivious tree outperforms non-oblivious on classification

On Adult, the oblivious tree (AUC=0.8285) substantially outperforms the
non-oblivious tree (AUC=0.7209).  The shared-split constraint at each level
acts as an implicit regulariser: fewer independent degrees of freedom forces
the model to find globally informative splits rather than locally optimal ones.

### 2. Classification gap from LightGBM is larger than regression gap

Regression (M2): SBT RMSE within **3.6%** of LightGBM.
Classification (M3): SBT AUC **10%** below LightGBM (oblivious tree).

Root cause (hypothesis): The per-node gradient normalisation was calibrated for
MSE (`h_i = 1`).  In binary cross-entropy, `h_i = p_i(1 − p_i) ≤ 0.25`, so
`H_total ≈ 0.25 · N_node` rather than `N_node`.  This makes `norm_gain` ~4×
larger than in regression, effectively loosening the screening threshold beyond
its calibrated range.  Dedicated classification calibration of `(s_w, s_r)` or
normalising by `n_node` instead of `H_total` in classification mode is a clear
M4 / follow-up item.

### 3. Screening accept_rate in classification

| Mode | Dataset | accept_rate |
|------|---------|-------------|
| non-oblivious | Titanic | 25.5% |
| oblivious | Titanic | 23.5% |
| non-oblivious | Adult | 11.8% |
| oblivious | Adult | 11.7% |

Accept_rate ≈ 10–25% indicates screening is active (not degenerate).  Adult's
lower rate reflects the harder, larger dataset.

## Parameter calibration note

Classification and regression share `s_w=-2.0, s_r=-6.0`.  Because
`h_i = p(1−p)` in binary mode, `H_total` is systematically smaller, which
inflates `norm_gain`.  Dedicated classification defaults are a pending item for
M4 (`s_r` closer to −2 to −4 may be more appropriate).
