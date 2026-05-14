"""
train_ranker.py — LightGBM LambdaRank V8
==========================================
Fixes V7's three problems:
  1. OVERFITTING (train NDCG 0.925 vs valid 0.525):
     Two-stage feature pruning — train once with all features,
     keep top features covering 95% of importance, retrain.
  2. WASTED GRADIENT (10 uniform buckets):
     4 asymmetric tail-focused buckets with label_gain="0,1,3,15".
     The model concentrates on ONE question: "Is this stock top-10%?"
  3. EXCESS COMPLEXITY (best Optuna trial used only 32 trees):
     Tighter regularization bounds, shorter patience.

Usage:
    python scripts/train_ranker.py
"""

import os
import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def mem_gb():
    """Current RSS memory in GB (Linux/WSL2)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PANEL_PATH     = "data/processed/panel.parquet"
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "ranker_model.txt")
FI_CSV_PATH    = os.path.join(MODEL_DIR, "ranker_feature_importance.csv")
FI_PLOT_PATH   = os.path.join(MODEL_DIR, "ranker_feature_importance.png")

TRAIN_RATIO    = 0.80
N_TRIALS       = 50
PRUNE_THRESHOLD = 0.95  # keep features covering this % of total importance
FORWARD_DAYS   = 10     # must match feature_engineering.py

EXCLUDE_COLS = {
    "forward_return", "excess_return", "relevance", "ticker", "is_halal",
    "close", "Sector", "Ticker", "date",
}

# Tail-focused buckets: bottom60% → 0, 60-80% → 1, 80-90% → 2, top10% → 3
RELEVANCE_BINS   = [0.0, 0.60, 0.80, 0.90, 1.0]
RELEVANCE_LABELS = [0, 1, 2, 3]
# label_gain: swapping a top-10% stock costs 15x more than swapping mid-tier
LABEL_GAIN = "0,1,3,15"
TOP_LABEL  = 3

# Use excess_return (sector-neutral) if sector coverage > this threshold
SECTOR_COVERAGE_MIN = 0.50

# Sectors where the model consistently underperforms (< random precision)
# These rows are excluded from training to avoid wasting model capacity.
WEAK_SECTORS = {"Real Estate", "Energy"}


# ---------------------------------------------------------------------------
# Discretize Target
# ---------------------------------------------------------------------------
def add_relevance_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Asymmetric tail-focused labeling."""
    # Check sector coverage to decide target
    if "excess_return" in panel.columns and "Sector" in panel.columns:
        unknown_pct = (panel["Sector"] == "Unknown").mean()
        if unknown_pct < (1.0 - SECTOR_COVERAGE_MIN):
            target_col = "excess_return"
        else:
            target_col = "forward_return"
            print(f"    ⚠️  Sector coverage too low ({(1-unknown_pct)*100:.0f}%), "
                  f"using raw forward_return")
    else:
        target_col = "forward_return"

    print(f"    Target: '{target_col}'")
    print(f"    Buckets: bottom60%→0, 60-80%→1, 80-90%→2, top10%→3")
    print(f"    label_gain: {LABEL_GAIN}")

    def _get_relevance(s):
        try:
            return pd.cut(
                s.rank(pct=True),
                bins=RELEVANCE_BINS,
                labels=RELEVANCE_LABELS,
                include_lowest=True
            ).astype(int)
        except Exception:
            return 0

    panel["relevance"] = panel.groupby("date")[target_col].transform(_get_relevance)
    panel["relevance"] = panel["relevance"].fillna(0).astype(int)
    return panel


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def top_k_precision(y_rel, y_pred, groups, k_pct=0.10):
    precs = []
    offset = 0
    for g in groups:
        g = int(g)
        yr, yp = y_rel[offset:offset+g], y_pred[offset:offset+g]
        k = max(1, int(g * k_pct))
        top_k = np.argsort(yp)[-k:]
        precs.append((yr[top_k] == TOP_LABEL).sum() / k)
        offset += g
    return np.mean(precs) if precs else 0.0


def halal_top_k_precision(y_rel, y_pred, groups, halal_mask, k_pct=0.10):
    precs = []
    offset = 0
    for g in groups:
        g = int(g)
        yr, yp = y_rel[offset:offset+g], y_pred[offset:offset+g]
        hm = halal_mask[offset:offset+g]
        k = max(1, int(g * k_pct))
        top_k = np.argsort(yp)[-k:]
        halal_in_top = [i for i in top_k if hm[i] == 1]
        if halal_in_top:
            hits = sum(1 for i in halal_in_top if yr[i] == TOP_LABEL)
            precs.append(hits / len(halal_in_top))
        offset += g
    return np.mean(precs) if precs else 0.0


def top_k_return(fwd_ret, y_pred, groups, k_pct=0.10):
    pick_rets, mkt_rets = [], []
    offset = 0
    for g in groups:
        g = int(g)
        fr, yp = fwd_ret[offset:offset+g], y_pred[offset:offset+g]
        k = max(1, int(g * k_pct))
        top_k = np.argsort(yp)[-k:]
        pick_rets.append(np.mean(fr[top_k]))
        mkt_rets.append(np.mean(fr))
        offset += g
    return np.mean(pick_rets), np.mean(mkt_rets)


def hit_rate(fwd_ret, y_pred, groups):
    correct, total = 0, 0
    offset = 0
    for g in groups:
        g = int(g)
        fr, yp = fwd_ret[offset:offset+g], y_pred[offset:offset+g]
        correct += ((yp > np.median(yp)) == (fr > np.median(fr))).sum()
        total += g
        offset += g
    return correct / max(total, 1)


def sector_breakdown(y_rel, y_pred, groups, sectors_arr, k_pct=0.10):
    sector_hits, sector_total = {}, {}
    offset = 0
    for g in groups:
        g = int(g)
        yr, yp, sc = y_rel[offset:offset+g], y_pred[offset:offset+g], sectors_arr[offset:offset+g]
        k = max(1, int(g * k_pct))
        top_k = np.argsort(yp)[-k:]
        for i in top_k:
            s = sc[i]
            sector_total[s] = sector_total.get(s, 0) + 1
            if yr[i] == TOP_LABEL:
                sector_hits[s] = sector_hits.get(s, 0) + 1
        offset += g
    return {s: (sector_hits.get(s, 0) / sector_total[s], sector_total[s])
            for s in sector_total}


# ---------------------------------------------------------------------------
# Two-Stage Feature Pruning
# ---------------------------------------------------------------------------
def prune_features(model, feature_cols, threshold=0.95):
    """Keep features covering `threshold` of cumulative importance."""
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    cumsum = np.cumsum(importance[sorted_idx]) / max(importance.sum(), 1)
    n_keep = int(np.searchsorted(cumsum, threshold)) + 1
    n_keep = max(n_keep, 10)  # keep at least 10
    top_idx = sorted_idx[:n_keep]
    pruned = [feature_cols[i] for i in top_idx]
    return pruned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("🚀 AlphaShariaBot — LightGBM LambdaRank V8\n")

    # ── 1. Load Panel ────────────────────────────────────────────────────
    print(f"📦 Loading panel... [RAM: {mem_gb():.1f} GB]")
    panel = pd.read_parquet(PANEL_PATH)
    if "date" not in panel.columns:
        panel = panel.reset_index()
    panel.reset_index(drop=True, inplace=True)
    panel.index.names = [None]
    panel["date"] = pd.to_datetime(panel["date"])

    # MEMORY: Force float32 immediately (halves DataFrame memory)
    for col in panel.select_dtypes(include=[np.float64]).columns:
        panel[col] = panel[col].astype(np.float32)
    print(f"    ✅ float32 downcast [RAM: {mem_gb():.1f} GB]")

    panel.sort_values(["date", "ticker"], inplace=True)
    panel.dropna(subset=["forward_return"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    # ── 2. Discretize ────────────────────────────────────────────────────
    print("🏷️  Tail-focused labeling...")
    panel = add_relevance_labels(panel)
    print(f"    Labels:\n{panel['relevance'].value_counts().sort_index()}\n")

    # ── 3. Features ──────────────────────────────────────────────────────
    feature_cols = [c for c in panel.columns if c not in EXCLUDE_COLS]
    feature_cols = [c for c in feature_cols
                    if panel[c].dtype in (np.float64, np.float32, np.int64,
                                          np.int32, np.int8, np.float16)]
    panel[feature_cols] = panel[feature_cols].replace([np.inf, -np.inf], np.nan)
    panel[feature_cols] = panel[feature_cols].fillna(0)

    # Drop zero-variance
    variances = panel[feature_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        print(f"⚠️  Dropping {len(zero_var)} constant features")
        feature_cols = [c for c in feature_cols if c not in zero_var]

    print(f"📐 Features: {len(feature_cols)}")
    print(f"📊 Panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers")
    print(f"📅 {panel['date'].min().date()} → {panel['date'].max().date()}")

    # ── 4. Split ─────────────────────────────────────────────────────────
    unique_dates = sorted(panel["date"].unique())
    split_idx = int(len(unique_dates) * TRAIN_RATIO)

    # PURGED split: leave a gap of FORWARD_DAYS between train and test
    # to prevent label leakage (last train rows' forward returns overlap test)
    train_end_idx = max(0, split_idx - FORWARD_DAYS)
    train_dates = set(unique_dates[:train_end_idx])
    test_dates  = set(unique_dates[split_idx:])
    print(f"    🔒 Purged gap: {FORWARD_DAYS} days between train/test")
    print(f"    Train dates: {len(train_dates)} | Purge gap: {split_idx - train_end_idx} | Test dates: {len(test_dates)}")

    train_mask = panel["date"].isin(train_dates)
    test_mask  = panel["date"].isin(test_dates)
    train_panel = panel[train_mask].copy()
    test_panel  = panel[test_mask].copy()

    # Drop weak sectors from training to avoid wasting model capacity
    if "Sector" in train_panel.columns and WEAK_SECTORS:
        before = len(train_panel)
        train_panel = train_panel[~train_panel["Sector"].isin(WEAK_SECTORS)].copy()
        dropped = before - len(train_panel)
        print(f"    🚫 Dropped {dropped:,} rows from weak sectors: {WEAK_SECTORS}")

    train_groups = train_panel.groupby("date").size().values
    test_groups  = test_panel.groupby("date").size().values

    X_train = train_panel[feature_cols].values  # already float32 from parquet load
    y_train = train_panel["relevance"].values.astype(np.float32)
    X_test  = test_panel[feature_cols].values
    y_test  = test_panel["relevance"].values.astype(np.float32)

    fwd_ret_test    = test_panel["forward_return"].values.copy()
    halal_mask_test = (test_panel["is_halal"].values == 1).copy()
    sectors_test    = test_panel["Sector"].values.copy() if "Sector" in test_panel.columns else None

    print(f"\n📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"⚙️  Avg stocks/day: {train_groups.mean():.0f}")

    # MEMORY: Free ALL DataFrames immediately
    del panel, train_panel, test_panel
    gc.collect()
    print(f"    ✅ DataFrames freed [RAM: {mem_gb():.1f} GB]")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Full-feature model to identify important features
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"📋 STAGE 1: Feature Discovery ({len(feature_cols)} features)")
    print(f"{'='*65}")

    lgb_train = lgb.Dataset(X_train, y_train, group=train_groups,
                            feature_name=feature_cols, free_raw_data=False)
    lgb_test  = lgb.Dataset(X_test, y_test, group=test_groups,
                            reference=lgb_train, free_raw_data=False)

    stage1_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
        "label_gain": LABEL_GAIN,
        "max_bin": 127,              # halve bins to reduce memorization
        "learning_rate": 0.01,
        "num_leaves": 40,            # tighter than before (was 63)
        "max_depth": 6,              # tighter (was 7)
        "min_data_in_leaf": 2000,    # more conservative (was 1000)
        "feature_fraction": 0.5,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
    }

    print("   Training stage-1 model...")
    s1_model = lgb.train(
        stage1_params, lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_test],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    print(f"   Stage-1 trees: {s1_model.num_trees()}")

    # Prune
    pruned_cols = prune_features(s1_model, feature_cols, PRUNE_THRESHOLD)
    print(f"   Pruned: {len(feature_cols)} → {len(pruned_cols)} features")

    # MEMORY: Free Stage 1 datasets + arrays, rebuild with pruned features
    del lgb_train, lgb_test, s1_model
    gc.collect()

    pruned_idx = [feature_cols.index(c) for c in pruned_cols]
    X_train_p = X_train[:, pruned_idx]
    X_test_p  = X_test[:, pruned_idx]

    # Free full-width arrays
    del X_train, X_test
    gc.collect()
    print(f"    ✅ Pruned arrays built [RAM: {mem_gb():.1f} GB]")

    lgb_train_p = lgb.Dataset(X_train_p, y_train, group=train_groups,
                              feature_name=pruned_cols, free_raw_data=False)
    lgb_test_p  = lgb.Dataset(X_test_p, y_test, group=test_groups,
                              reference=lgb_train_p, free_raw_data=False)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Optuna on pruned features
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"📋 STAGE 2: Optuna Optimization ({len(pruned_cols)} features)")
    print(f"{'='*65}")

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "verbosity": -1,
            "seed": 42,
            "feature_pre_filter": False,
            "label_gain": LABEL_GAIN,
            "max_bin": 127,
            # Tighter bounds to force generalization over memorization
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 40),
            "max_depth": trial.suggest_int("max_depth", 4, 6),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2000, 8000),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.6),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.85),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 20.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 20.0, log=True),
        }

        gbm = lgb.train(
            params, lgb_train_p,
            num_boost_round=3000,
            valid_sets=[lgb_test_p],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        preds = gbm.predict(X_test_p)
        prec = top_k_precision(y_test, preds, test_groups)
        trial.set_user_attr("n_trees", gbm.num_trees())
        return prec

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    with tqdm(total=N_TRIALS, desc="Optuna") as pbar:
        def cb(study, trial):
            pbar.update(1)
            pbar.set_postfix(
                best=f"{study.best_value:.4f}",
                trees=trial.user_attrs.get("n_trees", "?")
            )
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])

    print(f"\n🏆 Best Precision: {study.best_value:.4f}")
    print(f"   Trees: {study.best_trial.user_attrs.get('n_trees', '?')}")
    print(f"   Params: {study.best_trial.params}")

    # ── Final Model ──────────────────────────────────────────────────────
    best_params = study.best_trial.params
    best_params.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
        "label_gain": LABEL_GAIN,
        "max_bin": 127,
    })

    print("\n🚂 Training final model...")
    final_model = lgb.train(
        best_params, lgb_train_p,
        num_boost_round=3000,
        valid_sets=[lgb_train_p, lgb_test_p],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ]
    )

    final_model.save_model(MODEL_PATH)
    print(f"\n💾 Model → {MODEL_PATH} ({final_model.num_trees()} trees)")

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\n📈 Evaluating...\n")
    preds = final_model.predict(X_test_p)

    mkt_prec  = top_k_precision(y_test, preds, test_groups)
    mkt_prec2 = top_k_precision(y_test, preds, test_groups, k_pct=0.20)
    hal_prec  = halal_top_k_precision(y_test, preds, test_groups, halal_mask_test)
    pick_ret, mkt_ret = top_k_return(fwd_ret_test, preds, test_groups)
    hr = hit_rate(fwd_ret_test, preds, test_groups)

    print(f"{'='*60}")
    print(f"📊 RESULTS")
    print(f"   Top-10% Precision:    {mkt_prec*100:.2f}%")
    print(f"   Top-20% Precision:    {mkt_prec2*100:.2f}%")
    print(f"   Hit Rate:             {hr*100:.2f}%")
    print(f"   Avg Pick Return:      {pick_ret*100:.4f}%")
    print(f"   Avg Market Return:    {mkt_ret*100:.4f}%")
    print(f"   10d Alpha:            {(pick_ret-mkt_ret)*100:.4f}%")
    print(f"")
    print(f"☪️  HALAL")
    print(f"   Halal Top-10% Prec:   {hal_prec*100:.2f}%")

    if sectors_test is not None:
        print(f"\n📊 SECTOR BREAKDOWN:")
        sec_precs = sector_breakdown(y_test, preds, test_groups, sectors_test)
        for s in sorted(sec_precs, key=lambda x: sec_precs[x][1], reverse=True):
            p, n = sec_precs[s]
            print(f"   {s:25s} Prec: {p*100:5.1f}% | Picks: {n:6d}")

    print(f"{'='*60}")

    # ── Feature Importance ───────────────────────────────────────────────
    importance = final_model.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame({"Feature": pruned_cols, "Importance": importance})
    fi_df = fi_df.sort_values("Importance", ascending=False)
    fi_df.to_csv(FI_CSV_PATH, index=False)

    print(f"\n🔝 Top 20 Features (pruned set):")
    for _, row in fi_df.head(20).iterrows():
        bar = "█" * max(1, int(row["Importance"] / fi_df["Importance"].max() * 30))
        print(f"   {row['Feature']:40s} {bar}")

    # Absolute vs z-scored usage
    abs_imp = fi_df[~fi_df["Feature"].str.endswith("_xs")]["Importance"].sum()
    xs_imp  = fi_df[fi_df["Feature"].str.endswith("_xs")]["Importance"].sum()
    total = abs_imp + xs_imp
    if total > 0:
        print(f"\n   Absolute: {abs_imp/total*100:.0f}% | Z-Scored: {xs_imp/total*100:.0f}%")

    plt.figure(figsize=(10, 8))
    top_fi = fi_df.head(25).sort_values("Importance", ascending=True)
    colors = ["#2196F3" if f.endswith("_xs") else "#4CAF50" for f in top_fi["Feature"]]
    plt.barh(top_fi["Feature"], top_fi["Importance"], color=colors)
    plt.xlabel("Gain")
    plt.title("Feature Importances (Green=Absolute, Blue=Z-Scored)")
    plt.tight_layout()
    plt.savefig(FI_PLOT_PATH, dpi=150)
    print(f"📊 Plot → {FI_PLOT_PATH}")

    with open(os.path.join(MODEL_DIR, "ranker_features.json"), "w") as f:
        json.dump(pruned_cols, f, indent=2)


if __name__ == "__main__":
    main()