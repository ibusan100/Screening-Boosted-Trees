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
pip install screening-boosted-trees          # NumPy reference only
pip install "screening-boosted-trees[triton]" # + GPU Triton kernels
```

## Quick start

```python
from sbt import ScreeningBooster
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = ScreeningBooster(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_tr, y_tr.astype("float32"))
preds = model.predict(X_te)
rmse = float(np.sqrt(np.mean((y_te - preds) ** 2)))
print(f"RMSE: {rmse:.4f}")
print(f"Mean accept_rate: {model.mean_accept_rate():.1%}")
```

## Screening transform

```
raw_gain  = G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_total²/(H_total+λ)
norm_gain = raw_gain / H_total          # N-invariant normalisation
s         = 1 − exp(−norm_gain / τ)    # bounded similarity ∈ [0, 1)
ρ         = max(1 − r·(1−s), 0)²       # Trim-and-Square
```

Parameters `τ = exp(s_w) + ε` and `r = exp(s_r) + 1` are stored as
log-scale scalars and will become learnable in a future release.

## Status

| Milestone | Status |
|-----------|--------|
| M0: NumPy reference + Triton skeleton | ✅ |
| M1: Single regression tree + diagnostics | ✅ |
| M2: Triton numerical consistency + LightGBM/XGBoost benchmark | ✅ |
| M3: Oblivious tree + boosting loop optimisation | 🔜 |

## License

MIT © 2026 ibusan100
