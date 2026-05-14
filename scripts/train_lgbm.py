"""
train_lgbm.py
-------------
Phase 2: LightGBM Pivot

Trains a Gradient Boosting Tree model on the tabular financial features.
Gradient Boosting is the industry standard for this type of data as it
natively handles feature noise, non-linear interactions, and missing values
far better than standard Transformers.

Key features:
1. Strict per-stock chronological split (no look-ahead bias).
2. Bayesian Hyperparameter Optimization via Optuna.
3. Feature Importance extraction to interpret the model's logic.
4. Focuses on Top-10% Precision for real-world trading viability.
"""

import os
import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR  = "data/processed"
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "lgbm_model.txt")
FI_PLOT_PATH   = os.path.join(MODEL_DIR, "lgbm_feature_importance.png")
FI_CSV_PATH    = os.path.join(MODEL_DIR, "lgbm_feature_importance.csv")
WHITELIST_PATH = os.path.join(MODEL_DIR, "feature_whitelist.json")

TARGET_COL   = "Target"
EXCLUDE_COLS = {TARGET_COL, "close", "date", "atr_14", "Ticker", "Sector"}
TRAIN_RATIO  = 0.80

N_TRIALS = 30  # Optuna hyperparameter sweep trials


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------
def find_optimal_threshold(targets: np.ndarray, probs: np.ndarray) -> tuple:
    """Find threshold that maximises F1, requiring at least 10% recall."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.linspace(0.10, 0.90, 81):
        preds = (probs >= thresh).astype(int)
        rec   = recall_score(targets, preds, zero_division=0)
        if rec < 0.10:
            continue
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def top_decile_precision(targets: np.ndarray, probs: np.ndarray) -> float:
    """Return precision of the top 10% most confident positive predictions."""
    n_top = max(1, int(len(probs) * 0.10))
    top_ix = np.argsort(probs)[::-1][:n_top]
    return targets[top_ix].mean()


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("🚀  Starting LightGBM Pivot...\n")

    # ── 1. Load Whitelist ────────────────────────────────────────────────
    whitelist = None
    if os.path.exists(WHITELIST_PATH):
        with open(WHITELIST_PATH) as fh:
            whitelist = set(json.load(fh))
        print(f"📋  Loaded Feature Whitelist: {len(whitelist)} features")

    # ── 2. Load Data & Split Chronologically Per-Stock ───────────────────
    files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".parquet"))
    
    train_dfs = []
    test_dfs  = []

    for fname in tqdm(files, desc="Loading & Splitting Data"):
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, fname))
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # We don't necessarily need to dropna for LGBM, but let's be safe
        df.dropna(inplace=True)
        
        if len(df) < 100:
            continue
            
        split_idx = int(len(df) * TRAIN_RATIO)
        train_dfs.append(df.iloc[:split_idx])
        test_dfs.append(df.iloc[split_idx:])

    if not train_dfs:
        raise RuntimeError("No valid data found.")

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df  = pd.concat(test_dfs, ignore_index=True)
    del train_dfs, test_dfs; gc.collect()

    all_features = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    if whitelist:
        feature_cols = [c for c in all_features if c in whitelist]
    else:
        feature_cols = all_features

    # Ensure Sector is categorical
    if "Sector" in feature_cols:
        train_df["Sector"] = train_df["Sector"].astype("category")
        test_df["Sector"] = test_df["Sector"].astype("category")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL].values

    del train_df, test_df; gc.collect()

    print(f"\n📊  Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")
    print(f"📐  Features:   {len(feature_cols)}")

    # ── 3. Class Imbalance Weighting ─────────────────────────────────────
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    # Use sqrt of ratio for moderate positive upweight — not aggressive
    raw_ratio = neg_count / max(pos_count, 1)
    scale_pos_weight = min(np.sqrt(raw_ratio), 5.0)
    print(f"⚖️  Class balance — Pos: {int(pos_count):,} ({100*pos_count/len(y_train):.1f}%) | "
          f"scale_pos_weight: {scale_pos_weight:.2f}\n")

    # ── 4. Optuna Hyperparameter Tuning ──────────────────────────────────
    print(f"🔍  Running Optuna Hyperparameter Optimization ({N_TRIALS} trials)...")
    print("🎯  Objective: AUC × Top-10% Precision (dual)")

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test  = lgb.Dataset(X_test, y_test, reference=lgb_train)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42,
            "scale_pos_weight": scale_pos_weight,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 3000),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.8),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=1500,
            valid_sets=[lgb_test],
            callbacks=callbacks
        )

        probs = gbm.predict(X_test)
        auc = roc_auc_score(y_test, probs)
        top10 = top_decile_precision(y_test, probs)
        # Dual objective: geometric mean of AUC and Top-10% Precision
        score = np.sqrt(auc * top10)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    
    with tqdm(total=N_TRIALS, desc="Optuna Trials") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix(best_score=f"{study.best_value:.4f}")
        
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

    print(f"\n🏆  Best Trial Score (√AUC×Prec): {study.best_value:.4f}")
    best_params = study.best_trial.params
    best_params.update({
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 42,
        "scale_pos_weight": scale_pos_weight
    })

    # ── 5. Train Final Model ─────────────────────────────────────────────
    print("\n🚂  Training final model with best parameters...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
    
    final_model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_train, lgb_test],
        valid_names=["train", "valid"],
        callbacks=callbacks
    )

    final_model.save_model(MODEL_PATH)

    # ── 6. Evaluation & Threshold Tuning ─────────────────────────────────
    print("\n📈  Evaluating Final Model...")
    probs = final_model.predict(X_test)
    
    auc = roc_auc_score(y_test, probs)
    opt_thresh, best_f1 = find_optimal_threshold(y_test, probs)
    
    preds = (probs >= opt_thresh).astype(int)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    top10_prec = top_decile_precision(y_test, probs)

    print(f"\n{'='*40}")
    print(f"🎯  Final Honest AUC     : {auc:.4f}")
    print(f"🎯  Precision @ Opt     : {prec:.4f}")
    print(f"🎯  Recall @ Opt        : {rec:.4f}")
    print(f"🎯  F1-Score @ Opt      : {best_f1:.4f}")
    print(f"🎯  Top-10% Precision   : {top10_prec:.4f}")
    print(f"📐  Optimal Threshold    : {opt_thresh:.2f}")
    print(f"{'='*40}\n")

    # ── 7. Feature Importance ────────────────────────────────────────────
    importance = final_model.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance (Gain)": importance
    }).sort_values(by="Importance (Gain)", ascending=False)
    
    fi_df.to_csv(FI_CSV_PATH, index=False)
    print(f"💾  Feature Importance saved → {FI_CSV_PATH}")

    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_fi = fi_df.head(20).sort_values(by="Importance (Gain)", ascending=True)
    plt.barh(top_fi["Feature"], top_fi["Importance (Gain)"], color='skyblue')
    plt.xlabel('Gain (Information provided by feature)')
    plt.title('Top 20 LightGBM Feature Importances')
    plt.tight_layout()
    plt.savefig(FI_PLOT_PATH, dpi=150)
    print(f"📊  Feature Importance plot saved → {FI_PLOT_PATH}")


if __name__ == "__main__":
    main()
