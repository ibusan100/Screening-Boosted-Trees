# Screening Boosted Trees (`ibu_boost`)

An experimental port of the *screening* mechanism from
["Screening Is Enough"](https://arxiv.org/abs/2604.01178) (Nakanishi 2026) —
originally a Transformer attention primitive — into GBDT split selection.
NumPy reference implementation plus Triton GPU kernels, with end-to-end
numerical consistency tests.

**This is a research prototype**, not a drop-in replacement for LightGBM or
XGBoost. On the datasets tested so far it lags LightGBM by ~3.6% RMSE on
regression and ~10pp AUC on binary classification (see [Benchmarks](#benchmarks)
below). What it offers is a different way of asking the split-selection
question: instead of picking the relatively best candidate, each candidate
is scored by an absolute relevance value, and a node where all candidates
fall below threshold becomes a leaf — no `min_gain_to_split` heuristic.

## The idea, briefly

The Multiscreen paper observes that softmax attention can only express
*relative* relevance between keys: every key always receives some attention
mass, even when no key is genuinely relevant. The paper replaces softmax
with a *screening transform* that produces an absolute, bounded similarity
per (query, key) and then applies an explicit threshold.

GBDT split selection has the same shape. Standard gradient boosting picks
`argmax(gain)` over all (feature, bin) candidates and emits a split even
when the best gain is tiny. The screening transform, applied to gains
rather than attention scores, gives an analogous absolute rejection:

```
raw_gain  = G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_total²/(H_total+λ)
norm_gain = raw_gain / H_total          # N-invariant normalisation
s         = 1 − exp(−norm_gain / τ)    # bounded similarity ∈ [0, 1)
ρ         = max(1 − r·(1−s), 0)²       # Trim-and-Square
```

A node where `max(ρ) == 0` emits no split. Parameters
`τ = exp(s_w) + ε` and `r = exp(s_r) + 1` are stored in log space (see
`ScreeningParams`). They are currently fixed scalars; the closed-form
gradients `∂ρ/∂s_w` and `∂ρ/∂s_r` are already implemented in
`screening_split.py` for a future differentiable-surrogate experiment
(see [Open questions](#open-questions)).

Whether absolute-threshold rejection is genuinely useful for GBDT, or
whether it mostly recovers what `min_gain_to_split` already does with
extra steps, is the actual research question. The current results
suggest "useful for diagnostics and on regression, unclear on
classification" — see the benchmark and the h-normalisation write-up
below.

## Install

```bash
pip install ibu-boost            # NumPy reference only
pip install "ibu-boost[triton]"  # + GPU Triton kernels (CUDA)
```

**Windows GPU**: installs `triton-windows` automatically — same API as
upstream Triton.

## Quick start

```python
from ibu_boost import ScreeningBooster
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = ScreeningBooster(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    tree_type="oblivious",   # CatBoost-style symmetric splits
    device="cuda",           # requires [triton] extra
)
model.fit(X_tr, y_tr.astype("float32"))
preds = model.predict(X_te)
rmse = float(np.sqrt(np.mean((y_te - preds) ** 2)))
print(f"RMSE: {rmse:.4f}")
print(f"Mean accept_rate: {model.mean_accept_rate():.1%}")
```

## Benchmarks

All benchmarks use matched hyperparameters across models:
`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `lambda=1.0`,
`min_samples_leaf=20`, `num_bins=255`. 3-seed mean ± std on fixed 80/20
splits. Full per-milestone details in
[`benchmarks/REPORT.md`](benchmarks/REPORT.md),
[`REPORT_M3.md`](benchmarks/REPORT_M3.md),
[`REPORT_M4.md`](benchmarks/REPORT_M4.md),
[`REPORT_M6.md`](benchmarks/REPORT_M6.md).

### Regression — California Housing (N=20,640, F=8)

Non-oblivious tree, NumPy backend:

| Model | RMSE | Train time |
|-------|------|------------|
| **ScreeningBooster** | 0.4881 ± 0.0048 | 14.7 s |
| LightGBM 4.x | 0.4711 ± 0.0042 | 0.8 s |
| XGBoost 2.x | 0.4713 ± 0.0047 | 0.2 s |

SBT is within **~3.6% RMSE** of LightGBM. The training-time gap (~20×
slower) is mostly because SBT is pure NumPy/Python while LightGBM and
XGBoost are C++; see GPU section below.

### Binary classification — Adult / Census Income (N=48,842, F=14)

| Model | AUC-ROC | Accuracy | accept_rate |
|-------|---------|----------|-------------|
| **SBT oblivious** | 0.8261 ± 0.0089 | 0.804 | 5.9% |
| **SBT non-oblivious** | 0.7784 ± 0.0039 | 0.762 | 5.0% |
| LightGBM | 0.9282 ± 0.0006 | 0.873 | — |
| XGBoost | 0.9250 ± 0.0006 | 0.871 | — |

**SBT is ~10 percentage points AUC behind LightGBM on classification.**
This is an unresolved gap and the largest open problem in the project.
The shared `(s_w, s_r)` defaults were calibrated on regression; one
likely contributor is that classification needs different defaults (see
the h-normalisation write-up below for a related fix that partially
closed an earlier, larger gap).

### GPU speedup — California Housing, oblivious mode

| Backend | RMSE | Train time | Speedup |
|---------|------|------------|---------|
| NumPy (CPU) | 0.5286 ± 0.0039 | 5.34 s | 1× |
| **Triton (RTX 4060 Ti)** | **0.5286 ± 0.0039** | **1.70 s** | **3.15×** |

RMSE is numerically identical across backends (atol < 1e-3, enforced by
`tests/test_triton_consistency.py`).

### Kernel-level speedup

For a single node's full pipeline (histogram + missing stats + screening),
N=65536, F=8, B=255, 10% missing:

| Backend | Time / call | Speedup |
|---------|-------------|---------|
| NumPy (CPU) | 17.4 ms | 1× |
| **Triton (RTX 4060 Ti)** | 0.34 ms | **51×** |

The kernel-level speedup (51×) is much larger than the end-to-end
speedup (3.15×) because Python-side overhead — sample routing, leaf
value computation, tensor slicing — still runs on CPU. See
[`REPORT_M6.md`](benchmarks/REPORT_M6.md) for the full breakdown.

## The h-normalisation story (M3 → M4)

Included here as a write-up because it changed the implementation
substantially and illustrates how screening interacts with the loss
function.

Initial classification results (M3) were poor: non-oblivious AUC on
Adult was 0.7209. Investigation revealed that the screening transform's
per-node normalisation by `H_total` had been calibrated for MSE, where
`h_i = 1` and therefore `H_total = N_node`. In binary cross-entropy,
`h_i = p(1−p) ≤ 0.25`, so `H_total ≈ 0.25 × N_node` — about 4× smaller
than the calibration assumed. This pushed `norm_gain` ~4× higher than
its calibrated range, effectively loosening the screening threshold
beyond its intended operating point.

**Fix**: normalise `h` per node to mean = 1 before computing `H_total`.

Result on Adult, non-oblivious tree:

| | AUC | Std across seeds |
|---|---|---|
| M3 (no h normalisation) | 0.7209 | 0.0232 |
| M4 (per-node h mean = 1) | **0.7784** | **0.0039** |
| Delta | **+5.75pp** | **6× more stable** |

Oblivious tree AUC didn't move, suggesting per-level histogram
aggregation already partially absorbed the issue.

The remaining ~10pp gap to LightGBM after this fix is the main open
problem (see below).

## Key features

- **Absolute split rejection**: `max(ρ) == 0` → node becomes a leaf
  automatically. No `min_gain_to_split` hyperparameter, although
  `(s_w, s_r)` are themselves tunable and function as soft equivalents.
- **Oblivious tree mode** (`tree_type="oblivious"`): CatBoost-style
  symmetric splits — all nodes at a depth share one `(feature, bin)`.
  Faster GPU aggregation; better AUC on classification in current
  experiments.
- **Non-oblivious mode** (`tree_type="non_oblivious"`): each node
  selects its best split independently. Standard GBDT structure.
- **Missing-value handling**: XGBoost-style learned default direction
  per split — each candidate evaluates both miss-left and miss-right,
  picks the higher gain.
- **Binary classification** (`objective="binary"`): log-loss with
  sigmoid output.
- **Screening diagnostics**: `model.mean_accept_rate()` reports the
  fraction of split candidates accepted. Useful for catching
  over-rejection (all splits killed → tree degenerates to root) and
  under-rejection (screening is a no-op).
- **ScreeningParamSearch**: K-fold grid search over `(s_w, s_r)`.
  Because these parameters enter only post-histogram, tuning them
  requires no kernel changes.
- **Triton GPU kernels**: fused histogram scatter + screening
  transform; batched multi-node dispatch; full GPU pipeline with
  `X`, `g`, `h` pre-loaded and on-device gradient normalisation.

## Parameters

```python
ScreeningBooster(
    n_estimators    = 100,          # boosting rounds
    learning_rate   = 0.1,
    max_depth       = 6,
    min_samples_leaf = 20,
    num_bins        = 255,          # quantile bins per feature
    params          = ScreeningParams(s_w=-2.0, s_r=-6.0, lam=1.0),
    tree_type       = "oblivious",  # or "non_oblivious"
    objective       = "regression", # or "binary"
    device          = "cpu",        # or "cuda"
)
```

## Open questions

These are unresolved and are the natural next directions.

1. **The ~10pp AUC gap on Adult.** After fixing h normalisation (M4)
   the gap is much smaller and seed-to-seed variance dropped 6×, but
   the gap to LightGBM is still wide. Whether it closes with
   classification-specific `(s_w, s_r)` defaults (the current defaults
   were calibrated on regression), with a different normalisation
   choice for cross-entropy, or whether it reflects a more fundamental
   limitation of absolute-threshold split selection in this regime,
   is not known.

2. **Behaviour on larger / harder tabular benchmarks.** All current
   experiments are on small-to-medium datasets — California Housing
   (20k), Adult (48k), Titanic (1.3k). Larger and more diverse
   benchmarks such as Higgs (11M), KDD, and high-cardinality
   recommender-style data are untested. Both accuracy and the 51×
   kernel speedup are likely to behave differently at those scales.

3. **Learnable `(s_w, s_r)`.** The closed-form gradients
   `∂ρ/∂s_w` and `∂ρ/∂s_r` are already implemented in
   `screening_split.py` as "Phase 3 groundwork" but currently unused.
   Connecting them to a differentiable surrogate loss so that
   `(s_w, s_r)` update during training — rather than being grid-
   searched — is the next experiment.

Contributions, criticism, and counter-experiments are all welcome.

## Milestones

| Milestone | Status |
|-----------|--------|
| M0: NumPy reference + Triton skeleton | ✅ |
| M1: Single regression tree + diagnostics | ✅ |
| M2: Triton numerical consistency + LightGBM/XGBoost benchmark | ✅ |
| M3: Oblivious tree + binary classification | ✅ |
| M4: h normalisation + ScreeningParamSearch (K-fold grid tuner) | ✅ |
| M5: Missing value handling (XGBoost-style default direction) | ✅ |
| M6: Triton GPU kernels (histogram + screening fused) | ✅ |
| M7: Batched multi-node GPU dispatch + full GPU pipeline (3.15× E2E) | ✅ |
| Phase 2: Learnable `(s_w, s_r)` via differentiable surrogate | 🔜 |

## License

MIT © 2026 ibu_3 (ibusan100)
