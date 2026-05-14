"""
feature_selector.py
--------------------
Ticket #02 — Feature Selection (Correlation Pruning)

Run this BEFORE train_model.py. It:
  1. Loads all processed parquets and samples a representative subset
  2. Computes the Spearman correlation matrix across all features
  3. Greedily drops features with |ρ| > 0.92 to any already-selected feature
     (keeps the higher-variance feature of each correlated pair)
  4. Saves the surviving feature list to models/feature_whitelist.json

Why Spearman (not Pearson)?
  Financial features are rarely normally distributed. Spearman is rank-based
  and captures any monotonic relationship, not just linear ones.

Why 0.92?
  Empirically, features with |ρ| > 0.92 carry essentially no independent
  information. 0.92 is a conservative threshold; 0.85 is more aggressive.

Usage:
    python scripts/feature_selector.py [--threshold 0.92] [--sample 50000]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR  = "data/processed"
MODEL_DIR      = "models"
WHITELIST_PATH = os.path.join(MODEL_DIR, "feature_whitelist.json")
TARGET_COL     = "Target"
EXCLUDE_COLS   = {TARGET_COL, "close", "date", "atr_14", "Sector", "Ticker"}
DEFAULT_THRESH = 0.92
DEFAULT_SAMPLE = 80_000   # rows to sample for correlation (speed vs accuracy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sample(processed_dir: str, max_rows: int) -> pd.DataFrame:
    """Load a random sample from all processed parquets for correlation analysis."""
    files = [f for f in os.listdir(processed_dir) if f.endswith(".parquet")]
    dfs   = []
    rows_per_file = max(1, max_rows // max(len(files), 1))

    for fname in tqdm(files, desc="Loading for feature selection"):
        try:
            df = pd.read_parquet(os.path.join(processed_dir, fname))
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            if len(df) > rows_per_file:
                df = df.sample(rows_per_file, random_state=42)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        raise RuntimeError(f"No parquet files found in {processed_dir}")
    return pd.concat(dfs, ignore_index=True)


def greedy_correlation_drop(df: pd.DataFrame, threshold: float) -> list:
    """
    Greedy correlation pruning:
      - Sort features by descending variance (prefer more informative features)
      - For each feature, keep it if it has |ρ| <= threshold with ALL
        already-kept features; otherwise drop it.

    Returns the list of kept feature names.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Sort by variance descending (keep high-variance features preferentially)
    variances = df[feature_cols].select_dtypes(include=[np.number]).var().sort_values(ascending=False)
    sorted_cols  = variances.index.tolist()

    print(f"\n📐  Computing Spearman correlation matrix for {len(sorted_cols)} features "
          f"on {len(df):,} rows...")
    print("    (This may take 1-3 minutes for 100+ features)\n")

    # Compute pairwise Spearman correlation
    corr_matrix, _ = spearmanr(df[sorted_cols].values)
    if not isinstance(corr_matrix, np.ndarray):
        # spearmanr returns a scalar when there are only 2 columns
        corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
    corr_df = pd.DataFrame(np.abs(corr_matrix), index=sorted_cols, columns=sorted_cols)

    # Greedy selection
    kept    = []
    dropped = []

    for col in tqdm(sorted_cols, desc="Pruning correlated features"):
        if not kept:
            kept.append(col)
            continue
        # Check max absolute correlation with already-kept features
        max_corr = corr_df.loc[col, kept].max()
        if max_corr <= threshold:
            kept.append(col)
        else:
            dropped.append((col, round(max_corr, 4)))

    return kept, dropped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AlphaShariaBot Feature Selector")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESH,
                        help="Spearman |ρ| threshold above which to drop (default: 0.92)")
    parser.add_argument("--sample",    type=int,   default=DEFAULT_SAMPLE,
                        help="Max rows to sample for correlation computation")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load sample
    df = load_sample(PROCESSED_DIR, args.sample)
    total_features = len([c for c in df.columns if c not in EXCLUDE_COLS])
    print(f"\n📊  Total features before pruning : {total_features}")
    print(f"🎯  Correlation threshold          : |ρ| > {args.threshold}")
    print(f"🔢  Sample size                    : {len(df):,} rows\n")

    # Run pruning
    kept, dropped = greedy_correlation_drop(df, args.threshold)

    print(f"\n{'='*60}")
    print(f"✅  Features kept    : {len(kept)}")
    print(f"❌  Features dropped : {len(dropped)}")
    print(f"📉  Reduction        : {total_features} → {len(kept)} "
          f"({100*(total_features-len(kept))/total_features:.1f}% removed)\n")

    if dropped:
        print("Dropped features (shown with max |ρ| to a kept feature):")
        for name, rho in sorted(dropped, key=lambda x: -x[1])[:20]:
            print(f"  DROP  {name:<45}  |ρ|={rho:.4f}")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped)-20} more")

    print(f"\nKept features ({len(kept)}):")
    for name in kept:
        print(f"  KEEP  {name}")

    # Save whitelist
    with open(WHITELIST_PATH, "w") as fh:
        json.dump(kept, fh, indent=2)

    print(f"\n💾  Whitelist saved → {WHITELIST_PATH}")
    print(f"\n▶️   Next step: python scripts/train_model.py")


if __name__ == "__main__":
    main()
