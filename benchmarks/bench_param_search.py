"""
ScreeningParamSearch on Adult dataset — find best (s_w, s_r) for binary classification.

Run:
    python benchmarks/bench_param_search.py
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from ibu_boost import ScreeningBooster, ScreeningParamSearch

SEEDS = [0, 1, 2]
N_EST = 100
MAX_DEPTH = 6
LR = 0.1
LAM = 1.0


def load_adult():
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.data.copy()
    y_raw = data.target
    y = (y_raw.astype(str).str.strip() == ">50K").astype(np.float32)
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    X = df.values.astype(np.float64)
    col_medians = np.nanmedian(X, axis=0)
    for f in range(X.shape[1]):
        bad = ~np.isfinite(X[:, f])
        X[bad, f] = col_medians[f]
    return X, y


def run_lgbm_auc(X_tr, y_tr, X_te, y_te):
    model = lgb.LGBMClassifier(
        n_estimators=N_EST, learning_rate=LR, max_depth=MAX_DEPTH,
        num_leaves=2 ** MAX_DEPTH - 1, reg_lambda=LAM,
        min_child_samples=20, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, probs))


def main():
    print("Loading Adult dataset...")
    X, y = load_adult()
    print(f"  N={len(X)}, F={X.shape[1]}")

    # ------------------------------------------------------------------ #
    # Step 1: ScreeningParamSearch on seed=0 train split                  #
    # ------------------------------------------------------------------ #
    X_tr0, X_te0, y_tr0, y_te0 = train_test_split(X, y, test_size=0.2, random_state=0)

    print("\n[Step 1] Grid search (s_w x s_r) -- 3-fold CV on train split (seed=0)")
    t0 = time.perf_counter()
    searcher = ScreeningParamSearch(
        s_w_grid=(-4.0, -2.0, 0.0),
        s_r_grid=(-6.0, -4.0, -2.0, 0.0),
        n_estimators=50,          # reduced for speed during search
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        min_samples_leaf=20,
        lam=LAM,
        cv=3,
        objective="binary",
        tree_type="oblivious",
        verbose=True,
    )
    searcher.fit(X_tr0, y_tr0)
    t1 = time.perf_counter()
    print(f"\nSearch completed in {t1-t0:.1f}s")
    print(f"Best params: s_w={searcher.best_params_.s_w}, s_r={searcher.best_params_.s_r}")
    print(f"Best CV log-loss: {searcher.best_score_:.4f}")
    print("\nFull results table:")
    print(searcher.results_table())

    best_params = searcher.best_params_

    # ------------------------------------------------------------------ #
    # Step 2: Final eval — best params vs default params vs LightGBM      #
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Final evaluation across 3 seeds")
    print(f"  Best params: s_w={best_params.s_w}, s_r={best_params.s_r}")
    print(f"  Default params: s_w=-2.0, s_r=-6.0")

    from ibu_boost.screening_split import ScreeningParams
    default_params = ScreeningParams(s_w=-2.0, s_r=-6.0, lam=LAM)

    aucs_best    = []
    aucs_default = []
    aucs_lgbm    = []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

        # SBT with best params
        m_best = ScreeningBooster(
            n_estimators=N_EST, learning_rate=LR, max_depth=MAX_DEPTH,
            min_samples_leaf=20, params=best_params,
            tree_type="oblivious", objective="binary",
        )
        m_best.fit(X_tr, y_tr)
        auc_best = float(roc_auc_score(y_te, m_best.predict_proba(X_te)))
        aucs_best.append(auc_best)

        # SBT with default params
        m_def = ScreeningBooster(
            n_estimators=N_EST, learning_rate=LR, max_depth=MAX_DEPTH,
            min_samples_leaf=20, params=default_params,
            tree_type="oblivious", objective="binary",
        )
        m_def.fit(X_tr, y_tr)
        auc_def = float(roc_auc_score(y_te, m_def.predict_proba(X_te)))
        aucs_default.append(auc_def)

        auc_lgbm = run_lgbm_auc(X_tr, y_tr, X_te, y_te)
        aucs_lgbm.append(auc_lgbm)

        print(f"  [seed={seed}] SBT-best={auc_best:.4f}  SBT-default={auc_def:.4f}  LightGBM={auc_lgbm:.4f}")

    print("\n=== Summary ===")
    print(f"  SBT (best s_w={best_params.s_w}, s_r={best_params.s_r})  AUC={np.mean(aucs_best):.4f}±{np.std(aucs_best):.4f}")
    print(f"  SBT (default s_w=-2.0, s_r=-6.0)   AUC={np.mean(aucs_default):.4f}±{np.std(aucs_default):.4f}")
    print(f"  LightGBM                            AUC={np.mean(aucs_lgbm):.4f}±{np.std(aucs_lgbm):.4f}")
    gap = np.mean(aucs_lgbm) - np.mean(aucs_best)
    print(f"\n  Gap (LightGBM − SBT best): {gap:.4f} ({gap*100:.2f} pp)")

    results = {
        "best_params": {"s_w": best_params.s_w, "s_r": best_params.s_r},
        "cv_results": searcher.cv_results_,
        "final_eval": {
            "SBT_best":    {"auc_mean": float(np.mean(aucs_best)),    "auc_std": float(np.std(aucs_best))},
            "SBT_default": {"auc_mean": float(np.mean(aucs_default)), "auc_std": float(np.std(aucs_default))},
            "LightGBM":    {"auc_mean": float(np.mean(aucs_lgbm)),    "auc_std": float(np.std(aucs_lgbm))},
        },
    }
    out = Path(__file__).parent / "results_param_search.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
