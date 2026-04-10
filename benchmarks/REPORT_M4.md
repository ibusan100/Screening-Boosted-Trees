# M4/M5 Benchmark Report — h normalisation + missing value handling

## Setup

| Item | Value |
|------|-------|
| Date | 2026-04-08 |
| Platform | Windows 11 Home 10.0.26200 |
| Python | 3.11 |
| sbt version | 0.1.0 (NumPy reference, no Triton) |
| Changes vs M3 | Per-node h normalisation (mean=1); XGBoost-style missing value default direction |
| Reproduce | `python benchmarks/bench_vs_baseline.py` and `python benchmarks/bench_classification.py` |

## Hyperparameters (matched across all models)

| Param | Value |
|-------|-------|
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| lambda (L2 reg) | 1.0 |
| min_samples_leaf | 20 |
| SBT s_w / s_r | −2.0 / −6.0 |

---

## Regression — California Housing

| Model | RMSE (mean ± std) | Train time | Notes |
|-------|-------------------|------------|-------|
| **ScreeningBooster** | **0.4881 ± 0.0048** | 14.7 ± 1.3 s | NumPy only |
| LightGBM 4.x | 0.4711 ± 0.0042 | 0.8 s | — |
| XGBoost 2.x | 0.4713 ± 0.0047 | 0.2 s | — |

No change from M2: h normalisation has no effect on regression (`h_i=1` so `H_total=N_node` already).
Gap from LightGBM remains **3.6%**.

---

## Classification — Titanic (N=1 309, F=6)

| Model | AUC-ROC (mean ± std) | Accuracy | Train time |
|-------|----------------------|----------|------------|
| **SBT non-oblivious** | 0.7607 ± 0.0815 | 0.733 | 0.38 s |
| **SBT oblivious** | 0.7629 ± 0.0865 | 0.733 | 0.58 s |
| LightGBM | 0.8689 ± 0.0183 | 0.824 | 0.54 s |
| XGBoost | 0.8704 ± 0.0183 | 0.810 | 0.13 s |

Titanic sample size (~1 300) makes 3-seed std too large for conclusive comparison.
Marginal improvement over M3 (0.7578/0.7599 → 0.7607/0.7629).

---

## Classification — Adult (N=48 842, F=14)  ← primary evaluation

| Model | AUC-ROC (mean ± std) | Accuracy | Train time | accept_rate |
|-------|----------------------|----------|------------|-------------|
| **SBT non-oblivious** | **0.7784 ± 0.0039** | 0.762 | 39.2 s | 5.0% |
| **SBT oblivious** | **0.8261 ± 0.0089** | 0.804 | 18.2 s | 5.9% |
| LightGBM | 0.9282 ± 0.0006 | 0.873 | 0.43 s | — |
| XGBoost | 0.9250 ± 0.0006 | 0.871 | 0.20 s | — |

### M3 → M4 delta (Adult)

| Model | M3 AUC | M4 AUC | Delta | M3 std | M4 std |
|-------|--------|--------|-------|--------|--------|
| SBT non-oblivious | 0.7209 | **0.7784** | **+5.75 pp** | 0.0232 | **0.0039** |
| SBT oblivious | 0.8285 | 0.8261 | −0.24 pp | 0.0199 | 0.0089 |

**h normalisation significantly improved non-oblivious classification** (+5.75 pp AUC),
and made both models far more stable across seeds (std halved or better).

### Root cause analysis

In M3, `h_i = p(1−p) ≤ 0.25` caused `H_total ≈ 0.25 · N_node`, inflating
`norm_gain` ~4× beyond its calibrated range and making the screening threshold
effectively too loose.  Normalising h per node to `mean=1` (M4) restores
`H_total → N_node`, putting `norm_gain` back in the [0, 1] range regardless
of objective.

### Remaining gap

SBT oblivious is still **~10 pp** below LightGBM (0.8261 vs 0.9282).
Primary causes:
1. **No s_w/s_r tuning for classification** — current defaults (`s_w=-2.0, s_r=-6.0`)
   were calibrated on regression. ScreeningParamSearch (M4) can close this gap.
2. **NumPy speed**: Adult takes 18–39 s vs LightGBM's 0.4 s.
   Triton acceleration (M6) will address throughput.

---

## Screening diagnostics — Adult classification

| Mode | accept_rate | Notes |
|------|-------------|-------|
| non-oblivious | 5.0% | More aggressive rejection than M3 (was 11.8%) |
| oblivious | 5.9% | More aggressive rejection than M3 (was 11.7%) |

Accept_rate dropped with h normalisation because `H_total` is now larger
(= `N_node`), which reduces `norm_gain` to its calibrated range and tightens
the implicit threshold. This is expected and healthy — the model is not over-fitting
to all splits.
