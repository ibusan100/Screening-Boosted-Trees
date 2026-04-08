"""
Benchmark: ScreeningBooster (non_oblivious + oblivious) vs LightGBM vs XGBoost
on binary classification datasets.

Datasets:
  - Titanic (seaborn / sklearn fetch_openml)   N≈891, F=6 engineered features
  - Adult Income (sklearn fetch_openml)         N≈48842, F=14 mixed features

Metrics: AUC-ROC, Accuracy, train time
Seeds  : 0, 1, 2

Run:
    python benchmarks/bench_classification.py
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sbt.booster import ScreeningBooster
from sbt.screening_split import ScreeningParams

SEEDS = [0, 1, 2]
N_EST = 100
MAX_DEPTH = 6
LR = 0.1
LAM = 1.0
SBT_PARAMS = ScreeningParams(s_w=-2.0, s_r=-6.0, lam=LAM)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_titanic():
    """Titanic survival prediction (binary).

    Source: OpenML dataset 40945 (Titanic: Machine Learning from Disaster).
    Features: Pclass, Sex, Age, SibSp, Parch, Fare  (6 after encoding).
    Missing Age/Fare filled with median; Sex ordinal-encoded.
    """
    data = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    feat_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    df = df[feat_cols + ["survived"]].dropna(subset=["survived"])
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())

    enc = OrdinalEncoder()
    df[["sex"]] = enc.fit_transform(df[["sex"]])

    X = df[feat_cols].values.astype(np.float64)
    y = df["survived"].astype(float).values.astype(np.float32)
    return X, y, "Titanic"


def load_adult():
    """Adult income (binary): predict >50K salary.

    Source: OpenML dataset 1590 (Adult / Census Income).
    Categorical columns ordinal-encoded; missing values (-1) replaced with col median.
    """
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    # data.data contains features only (no target); data.frame includes target column
    df = data.data.copy()

    y_raw = data.target
    y = (y_raw.astype(str).str.strip() == ">50K").astype(np.float32)

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

    X = df.values.astype(np.float64)
    # Replace any residual NaN / Inf
    col_medians = np.nanmedian(X, axis=0)
    for f in range(X.shape[1]):
        bad = ~np.isfinite(X[:, f])
        X[bad, f] = col_medians[f]

    return X, y, "Adult"


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------


def run_sbt(X_tr, y_tr, X_te, y_te, tree_type="non_oblivious"):
    model = ScreeningBooster(
        n_estimators=N_EST, learning_rate=LR, max_depth=MAX_DEPTH,
        min_samples_leaf=20, num_bins=255,
        params=SBT_PARAMS,
        tree_type=tree_type,
        objective="binary",
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    probs = model.predict_proba(X_te)
    auc = float(roc_auc_score(y_te, probs))
    acc = float(accuracy_score(y_te, (probs > 0.5).astype(int)))
    return auc, acc, t1 - t0, model.mean_accept_rate()


def run_lgbm(X_tr, y_tr, X_te, y_te):
    model = lgb.LGBMClassifier(
        n_estimators=N_EST, learning_rate=LR, max_depth=MAX_DEPTH,
        num_leaves=2 ** MAX_DEPTH - 1, reg_lambda=LAM,
        min_child_samples=20, verbose=-1,
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    probs = model.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, probs)), float(accuracy_score(y_te, (probs > 0.5).astype(int))), t1 - t0


def run_xgb(X_tr, y_tr, X_te, y_te):
    model = xgb.XGBClassifier(
        n_estimators=N_EST, learning_rate=LR, max_depth=MAX_DEPTH,
        reg_lambda=LAM, min_child_weight=20, verbosity=0,
        eval_metric="logloss",
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    probs = model.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, probs)), float(accuracy_score(y_te, (probs > 0.5).astype(int))), t1 - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def benchmark_dataset(X, y, name):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  (N={len(X)}, F={X.shape[1]})")
    print(f"{'='*60}")

    results = {
        "SBT_non_oblivious": {"auc": [], "acc": [], "time_s": [], "accept_rate": []},
        "SBT_oblivious":     {"auc": [], "acc": [], "time_s": [], "accept_rate": []},
        "LightGBM":          {"auc": [], "acc": [], "time_s": []},
        "XGBoost":           {"auc": [], "acc": [], "time_s": []},
    }

    for seed in SEEDS:
        print(f"\n[seed={seed}]")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

        for tree_type, key in [("non_oblivious", "SBT_non_oblivious"), ("oblivious", "SBT_oblivious")]:
            auc, acc, t, ar = run_sbt(X_tr, y_tr, X_te, y_te, tree_type=tree_type)
            results[key]["auc"].append(auc)
            results[key]["acc"].append(acc)
            results[key]["time_s"].append(t)
            results[key]["accept_rate"].append(ar)
            tag = "non-obliv" if tree_type == "non_oblivious" else "obliv    "
            print(f"  SBT({tag})  AUC={auc:.4f}  Acc={acc:.3f}  time={t:.2f}s  accept={ar:.1%}")

        lauc, lacc, lt = run_lgbm(X_tr, y_tr, X_te, y_te)
        results["LightGBM"]["auc"].append(lauc)
        results["LightGBM"]["acc"].append(lacc)
        results["LightGBM"]["time_s"].append(lt)
        print(f"  LightGBM           AUC={lauc:.4f}  Acc={lacc:.3f}  time={lt:.2f}s")

        xauc, xacc, xt = run_xgb(X_tr, y_tr, X_te, y_te)
        results["XGBoost"]["auc"].append(xauc)
        results["XGBoost"]["acc"].append(xacc)
        results["XGBoost"]["time_s"].append(xt)
        print(f"  XGBoost            AUC={xauc:.4f}  Acc={xacc:.3f}  time={xt:.2f}s")

    print(f"\n=== {name} Summary ===")
    stats = {}
    for mname, vals in results.items():
        auc_m = float(np.mean(vals["auc"]))
        auc_s = float(np.std(vals["auc"]))
        acc_m = float(np.mean(vals["acc"]))
        t_m   = float(np.mean(vals["time_s"]))
        ar_s  = f"  accept={np.mean(vals['accept_rate']):.1%}" if "accept_rate" in vals else ""
        print(f"  {mname:<22} AUC={auc_m:.4f}±{auc_s:.4f}  Acc={acc_m:.3f}  time={t_m:.2f}s{ar_s}")
        stats[mname] = {
            "auc_mean": auc_m, "auc_std": auc_s,
            "acc_mean": acc_m,
            "time_mean": t_m,
        }
        if "accept_rate" in vals:
            stats[mname]["accept_rate_mean"] = float(np.mean(vals["accept_rate"]))
    return stats


def main():
    all_stats = {}
    for loader in [load_titanic, load_adult]:
        X, y, name = loader()
        all_stats[name] = benchmark_dataset(X, y, name)

    out_path = Path(__file__).parent / "results_classification.json"
    out_path.write_text(json.dumps({
        "config": {
            "n_estimators": N_EST, "max_depth": MAX_DEPTH,
            "learning_rate": LR, "lam": LAM,
            "sbt_params": {"s_w": -2.0, "s_r": -6.0},
            "seeds": SEEDS,
        },
        "results": all_stats,
    }, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
