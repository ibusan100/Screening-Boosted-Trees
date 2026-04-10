# Screening Boosted Trees

Gradient-boosted decision trees where each candidate split is scored by an
**absolute relevance** value derived from the *screening* transform introduced
in ["Screening Is Enough"](https://arxiv.org/abs/2604.01178) (Nakanishi 2026).

Instead of selecting the *relatively* best split among all candidates, a
node with `max(ρ) == 0` emits **no split at all** — becoming a leaf without
an external `min_gain_to_split` heuristic. The acceptance threshold is
learnable.

## Install

```bash
pip install ibu-boost          # NumPy reference only
pip install "ibu-boost[triton]" # + GPU Triton kernels (CUDA)
```

**Windows GPU**: installs `triton-windows` automatically — same API as upstream Triton.

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

## GPU performance (RTX 4060 Ti, California Housing, 100 rounds)

| Backend | Train time | Speedup | RMSE |
|---------|-----------|---------|------|
| CPU (NumPy) | 5.34 s | 1x | 0.5286 ± 0.0039 |
| **CUDA (Triton)** | **1.70 s** | **3.15x** | **0.5286 ± 0.0039** |

Kernel-level speedup (N=65536, F=8, B=255): **51x** over NumPy.
RMSE is numerically identical across backends.

## Screening transform

```
raw_gain  = G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_total²/(H_total+λ)
norm_gain = raw_gain / H_total          # N-invariant normalisation
s         = 1 − exp(−norm_gain / τ)    # bounded similarity ∈ [0, 1)
ρ         = max(1 − r·(1−s), 0)²       # Trim-and-Square
```

Parameters `τ = exp(s_w) + ε` and `r = exp(s_r) + 1` are stored as
log-scale scalars (`ScreeningParams`). They will become learnable in a future release.

## Key features

- **Absolute split rejection**: `max(ρ) == 0` → node becomes leaf automatically.
  No `min_gain_to_split` hyperparameter required.
- **Oblivious tree mode** (`tree_type="oblivious"`): CatBoost-style symmetric
  splits — all nodes at a depth share one (feature, bin). Fast GPU aggregation.
- **Non-oblivious mode** (`tree_type="non_oblivious"`): each node independently
  selects its best split (standard GBDT structure).
- **Missing value handling**: XGBoost-style learned default direction per split.
- **Binary classification** (`objective="binary"`): log-loss with sigmoid output.
- **Screening diagnostics**: `model.mean_accept_rate()` shows fraction of splits
  accepted — a real-time indicator of over/under-rejection.
- **ScreeningParamSearch**: K-fold grid search over `(s_w, s_r)`.
- **Triton GPU kernels**: fused histogram scatter + screening transform; batched
  multi-node dispatch; full GPU pipeline (X/g/h pre-loaded, on-device normalisation).

## Parameters

```python
ScreeningBooster(
    n_estimators   = 100,          # boosting rounds
    learning_rate  = 0.1,
    max_depth      = 6,
    min_samples_leaf = 20,
    num_bins       = 255,          # quantile bins per feature
    params         = ScreeningParams(s_w=-2.0, s_r=-6.0, lam=1.0),
    tree_type      = "oblivious",  # or "non_oblivious"
    objective      = "regression", # or "binary"
    device         = "cpu",        # or "cuda"
)
```

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
| M7: Batched multi-node GPU dispatch + full GPU pipeline (3.15x E2E) | ✅ |
| Phase 2: Learnable `(s_w, s_r)` via differentiable surrogate | 🔜 |

## License

MIT © 2026 ibusan100
