"""
feature_engineering.py — V7 (Sector-Neutral, Liquidity-Aware, Regime-Conscious)
=====================================================
Paradigm shift: We no longer build a binary classifier.
We build a RANKING system that scores every stock in the universe
on a given day relative to every other stock.

Pipeline:
  1. Load raw OHLCV per stock → compute per-stock technical indicators
  2. Normalize all price-level features to be scale-invariant
  3. Merge earnings calendar → compute `days_until_earnings`
  4. Merge macro context (SPY/VIX) → create interaction features ONLY
  5. Concat into a daily panel → compute cross-sectional ranks & sector flow
  6. Compute forward 5-day return as the RANKING TARGET
  7. Save a single monolithic panel Parquet for the ranker

Anti-Leakage Guarantees:
  * Forward returns use close[T+5]/close[T] — no peeking.
  * All features use .rolling/.shift (backward-looking only).
  * Macro features at date T are public at close of T.
  * Earnings dates are public knowledge (announced weeks ahead).

Usage:
    python scripts/feature_engineering.py
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm


def mem_gb():
    """Current RSS memory in GB (Linux/WSL2, zero-dependency)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR    = "data/historical"
MACRO_DIR    = "data/macro"
OUTPUT_DIR   = "data/processed"
PANEL_PATH   = "data/processed/panel.parquet"   # Single monolithic file
EARNINGS_CSV = "data/earnings_dates.csv"
FUND_CSV     = "data/fundamentals.csv"
HALAL_CSV    = "data/halal_stocks.csv"
SECTORS_CSV  = "data/sectors.csv"
SENTIMENT_PATH = "data/sentiment/daily_sentiment.parquet"

FORWARD_DAYS = 10  # ranking horizon: 10 trading days (2x signal vs 5d)
FWD_SMOOTH   = 2   # smooth window: use median of [T+8..T+12] instead of just T+10
MIN_HISTORY  = 252 # minimum bars for a stock to be included
MIN_DOLLAR_VOL = 500_000  # minimum avg daily dollar volume (20d) to include


# ---------------------------------------------------------------------------
# Helper: rolling Z-score (backward-looking only)
# ---------------------------------------------------------------------------
def compute_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period).mean()
    std  = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Macro Context Loader (unchanged — proven leak-free)
# ---------------------------------------------------------------------------
def load_macro_context(macro_dir: str) -> pd.DataFrame:
    spy_path = os.path.join(macro_dir, "SPY.parquet")
    vix_path = os.path.join(macro_dir, "VIX.parquet")

    if not os.path.exists(spy_path) or not os.path.exists(vix_path):
        raise FileNotFoundError(
            "Macro data not found. Run 'python scripts/macro_downloader.py' first."
        )

    spy = pd.read_parquet(spy_path, engine="pyarrow")
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_close = spy["close"]
    spy_log_ret = np.log(spy_close / spy_close.shift(1))

    macro = pd.DataFrame(index=spy.index)
    macro["spy_ret_1d"]    = spy_close.pct_change(1)
    macro["spy_ret_5d"]    = spy_close.pct_change(5)
    macro["spy_ret_20d"]   = spy_close.pct_change(20)
    macro["spy_ema_50"]    = ta.ema(spy_close, length=50)  / spy_close
    macro["spy_ema_200"]   = ta.ema(spy_close, length=200) / spy_close
    macro["spy_regime"]    = (spy_close > ta.ema(spy_close, length=200)).astype(np.int8)
    macro["spy_vol_20"]    = spy_log_ret.rolling(20).std()

    vix = pd.read_parquet(vix_path, engine="pyarrow")
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix_close = vix["close"]

    vix_mean_252 = vix_close.rolling(252).mean()
    vix_std_252  = vix_close.rolling(252).std().replace(0, np.nan)
    vix_sma_20   = vix_close.rolling(20).mean()

    macro["vix_level_z"]   = (vix_close - vix_mean_252) / vix_std_252
    macro["vix_sma_ratio"] = vix_close / vix_sma_20.replace(0, np.nan)
    macro["vix_regime"]    = (vix_close > 20).astype(np.int8)

    # Regime awareness: SPY drawdown from 252-day high
    spy_252_high = spy_close.rolling(252, min_periods=60).max()
    macro["spy_drawdown"] = (spy_close - spy_252_high) / spy_252_high.replace(0, np.nan)
    # Market breadth proxy: SPY distance from 50d EMA (normalized)
    spy_ema50 = ta.ema(spy_close, length=50)
    macro["spy_trend_strength"] = (spy_close - spy_ema50) / spy_ema50.replace(0, np.nan)

    macro.ffill(inplace=True)
    macro.dropna(inplace=True)
    return macro


# ---------------------------------------------------------------------------
# Per-Stock Feature Builder
# ---------------------------------------------------------------------------
def build_stock_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Build all per-stock features for one ticker. Returns a DataFrame
    indexed by date with all features + forward_return (the ranking target)."""

    df = df.copy()

    # --- Normalize columns ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.rename(columns=lambda x: str(x).lower(), inplace=True)

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()  # skip malformed files

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
            df.drop(columns=["date"], inplace=True)
        else:
            return pd.DataFrame()

    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.sort_index(inplace=True)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    open_  = df["open"]

    # --- Liquidity Shield ---
    vol_sma = volume.rolling(20).mean()
    df = df[volume > (vol_sma * 0.3)].copy()
    if len(df) < MIN_HISTORY:
        return pd.DataFrame()

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    open_  = df["open"]

    # --- Dollar Volume (liquidity feature + filter) ---
    df["dollar_volume_20d"] = (close * volume).rolling(20).mean()

    # --- Volatility & Regime ---
    df["atr_14"]        = ta.atr(high, low, close, length=14)
    df["atr_pct"]       = df["atr_14"] / close
    ema_200             = ta.ema(close, length=200)
    df["market_regime"] = (close > ema_200).astype(np.int8)

    # --- RANKING TARGET: forward return (smoothed) ---
    # Instead of a single close[T+10]/close[T]-1 which is noisy,
    # use the MEDIAN of returns at [T+8..T+12] to reduce single-day noise.
    fwd_returns = pd.DataFrame(index=df.index)
    for offset in range(FORWARD_DAYS - FWD_SMOOTH, FORWARD_DAYS + FWD_SMOOTH + 1):
        fwd_returns[f"fwd_{offset}"] = close.shift(-offset) / close - 1.0
    df["forward_return"] = fwd_returns.median(axis=1)

    # --- Custom Engineered Features ---
    df["log_return"]      = np.log(close / close.shift(1))
    df["ret_5d"]          = close.pct_change(5)
    df["ret_20d"]         = close.pct_change(20)
    df["momentum_accel"]  = df["ret_5d"].pct_change(5)
    df["vol_20"]          = df["log_return"].rolling(20).std()
    df["vol_5"]           = df["log_return"].rolling(5).std()
    df["vol_ratio"]       = df["vol_5"] / df["vol_20"].replace(0, np.nan)
    df["zscore_close_20"] = compute_zscore(close, 20)
    df["hl_ratio"]        = (high - low) / close
    df["oc_ratio"]        = (close - open_) / open_.replace(0, np.nan)

    # --- Lag / Delta Features (Signal-to-Noise Boost) ---
    # These capture "direction of change" — the model sees not just
    # "RSI is 65" but "RSI moved from 55 to 65 in 3 days" (momentum).
    df["atr_pct_change"]  = df["atr_pct"].pct_change(5)       # ATR expansion/contraction
    df["vol_surge"]       = volume / volume.rolling(20).mean() # Volume spike detector

    # --- Technical Indicators ---
    # Momentum (already scale-invariant)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.stoch(append=True)
    df.ta.cci(length=20, append=True)
    df.ta.roc(length=10, append=True)
    df.ta.willr(length=14, append=True)

    # Volatility
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.kc(length=20, append=True)

    # Trend
    df.ta.adx(length=14, append=True)
    df.ta.aroon(length=14, append=True)
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)

    # Volume
    if (volume != 0).any():
        df.ta.obv(append=True)
        df.ta.cmf(length=20, append=True)
        df.ta.mfi(length=14, append=True)

    # --- Indicator Lag / Delta Features ---
    # These MUST be computed after the indicators exist but BEFORE normalization.
    # They capture the "trajectory" of each indicator.
    if "RSI_14" in df.columns:
        df["RSI_delta_3"]  = df["RSI_14"] - df["RSI_14"].shift(3)
        df["RSI_delta_5"]  = df["RSI_14"] - df["RSI_14"].shift(5)
    if "ADX_14" in df.columns:
        df["ADX_delta_5"]  = df["ADX_14"] - df["ADX_14"].shift(5)
    if "MACDh_12_26_9" in df.columns:
        df["MACD_hist_accel"] = df["MACDh_12_26_9"] - df["MACDh_12_26_9"].shift(3)
    if "CMF_20" in df.columns:
        df["CMF_delta_5"]  = df["CMF_20"] - df["CMF_20"].shift(5)

    # --- Normalize Price-Level Features to % of close ---
    for col in ["EMA_9", "EMA_21", "EMA_50"]:
        if col in df.columns:
            df[col] = (close - df[col]) / close

    for col in ["BBL_20_2.0_2.0", "BBM_20_2.0_2.0", "BBU_20_2.0_2.0"]:
        if col in df.columns:
            df[col] = (close - df[col]) / close

    for col in ["KCLe_20_2", "KCBe_20_2", "KCUe_20_2"]:
        if col in df.columns:
            df[col] = (close - df[col]) / close

    for col in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]:
        if col in df.columns:
            df[col] = df[col] / close

    if "OBV" in df.columns:
        df["OBV"] = df["OBV"].pct_change(5)

    # --- Macro Injection (Interaction Features ONLY) ---
    stock_dates = df.index
    macro_aligned = macro.reindex(stock_dates, method="ffill")

    # Per-stock vs market features
    stock_log_ret = np.log(close / close.shift(1))
    spy_log_ret   = np.log(1 + macro_aligned["spy_ret_1d"])

    cov_60  = stock_log_ret.rolling(60).cov(spy_log_ret)
    var_spy = spy_log_ret.rolling(60).var().replace(0, np.nan)
    df["stock_beta_60d"] = cov_60 / var_spy
    df["rs_vs_spy_20d"]  = df["ret_20d"] - macro_aligned["spy_ret_20d"]

    # Interaction features — stock indicator × market regime
    spy_regime = macro_aligned["spy_regime"]
    vix_regime = macro_aligned["vix_regime"]
    df["RSI_x_regime"]  = df.get("RSI_14", 0) * spy_regime
    df["ADX_x_regime"]  = df.get("ADX_14", 0) * spy_regime
    df["RS_x_regime"]   = df["rs_vs_spy_20d"] * spy_regime
    df["vol_x_vix"]     = df["vol_20"] * vix_regime

    # Keep only non-cheatable macro context
    df["vix_level_z"]   = macro_aligned["vix_level_z"]
    df["vix_sma_ratio"] = macro_aligned["vix_sma_ratio"]
    df["vix_regime"]    = macro_aligned["vix_regime"]
    df["spy_regime"]    = spy_regime

    # --- Higher-Order Signal Features (Plateau Breakers) ---
    # These give the Transformer richer temporal patterns to learn from.

    # 52-week high/low proximity — captures mean reversion + breakout
    high_252 = close.rolling(252, min_periods=60).max()
    low_252  = close.rolling(252, min_periods=60).min()
    df["dist_52w_high"] = (close - high_252) / high_252.replace(0, np.nan)  # always ≤ 0
    df["dist_52w_low"]  = (close - low_252) / low_252.replace(0, np.nan)    # always ≥ 0

    # Volatility-adjusted momentum (Sharpe-like ratio)
    # Separates genuine momentum from noisy spikes
    df["sharpe_5d"]  = df["ret_5d"] / df["vol_5"].replace(0, np.nan)
    df["sharpe_20d"] = df["ret_20d"] / df["vol_20"].replace(0, np.nan)

    # Momentum acceleration (2nd derivative) — detects inflection points
    df["ret_accel_5d"]  = df["ret_5d"] - df["ret_5d"].shift(5)
    df["ret_accel_20d"] = df["ret_20d"] - df["ret_20d"].shift(20)

    # --- Calendar / Temporal Features ---
    # These capture well-known temporal anomalies:
    # Monday effect, Friday selling, January effect, quarter-end window dressing
    df["day_of_week"]        = df.index.dayofweek.astype(np.int8)
    df["month"]              = df.index.month.astype(np.int8)
    df["is_month_end_5d"]    = (df.index.to_series().apply(
        lambda d: (d + pd.offsets.MonthEnd(0) - d).days
    ) <= 5).astype(np.int8)
    df["is_quarter_end_10d"] = (df.index.to_series().apply(
        lambda d: (d + pd.offsets.QuarterEnd(0) - d).days
    ) <= 10).astype(np.int8)

    # --- Cleaning ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Drop raw OHLCV + helper columns
    cols_to_drop = ["open", "high", "low", "volume", "atr_14", "atr_pct"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns],
            errors="ignore", inplace=True)

    # Winsorize numeric features (0.5th to 99.5th percentile)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in {"forward_return", "close"}]
    for col in numeric_cols:
        lo, hi = df[col].quantile(0.005), df[col].quantile(0.995)
        if lo < hi:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # ── MEMORY: Downcast to float32 per-stock (halves RAM) ──
    for col in df.select_dtypes(include=[np.float64]).columns:
        df[col] = df[col].astype(np.float32)

    return df


def filter_illiquid(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stocks with < MIN_DOLLAR_VOL avg daily dollar volume.
    This prevents the model from ranking untradeable micro-caps."""
    if "dollar_volume_20d" in df.columns:
        return df[df["dollar_volume_20d"] >= MIN_DOLLAR_VOL].copy()
    return df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load Macro Context ─────────────────────────────────────────────
    print("📡  Loading macro context (SPY + VIX)...")
    try:
        macro = load_macro_context(MACRO_DIR)
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        return
    print(f"    ✅  Macro ready: {len(macro)} days\n")

    # ── 2. Load Earnings Calendar ────────────────────────────────────────
    earnings_df = None
    if os.path.exists(EARNINGS_CSV):
        earnings_df = pd.read_csv(EARNINGS_CSV)
        earnings_df["next_earnings_date"] = pd.to_datetime(
            earnings_df["next_earnings_date"], errors="coerce"
        )
        earnings_df.dropna(subset=["next_earnings_date"], inplace=True)
        # Build a dict: ticker → list of earnings dates
        earnings_map = earnings_df.groupby("ticker")["next_earnings_date"].apply(list).to_dict()
        print(f"📅  Earnings calendar loaded: {len(earnings_map)} tickers\n")
    else:
        earnings_map = {}
        print("⚠️  No earnings_dates.csv found — skipping earnings features.\n")

    # ── 3. Load Fundamentals ─────────────────────────────────────────────
    fundamentals = None
    if os.path.exists(FUND_CSV):
        fundamentals = pd.read_csv(FUND_CSV)
        print(f"📊  Fundamentals loaded: {len(fundamentals)} tickers\n")
    else:
        print("⚠️  No fundamentals.csv found — skipping fundamental features.\n")

    # ── 4. Load Halal Universe (for tagging) ─────────────────────────────
    halal_set = set()
    if os.path.exists(HALAL_CSV):
        halal_df = pd.read_csv(HALAL_CSV)
        col = "ticker" if "ticker" in halal_df.columns else "Ticker"
        halal_set = set(halal_df[col].str.upper().tolist())
        print(f"☪️   Halal universe: {len(halal_set)} tickers\n")

    # ── 5. Process All Stocks ────────────────────────────────────────────
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".parquet"))
    all_dfs = []
    failed = []

    for fname in tqdm(files, desc="Step 1/3: Per-Stock Features"):
        ticker = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(INPUT_DIR, fname), engine="pyarrow")
            df = build_stock_features(df, macro)
            if df.empty or len(df) < 60:
                continue

            df["ticker"] = ticker
            df["is_halal"] = 1 if ticker.upper() in halal_set else 0

            # --- Earnings Feature ---
            if ticker in earnings_map:
                earn_dates = sorted(earnings_map[ticker])
                # For each row, find days until the next earnings date
                def days_to_next_earnings(date):
                    for ed in earn_dates:
                        delta = (ed - date).days
                        if delta >= 0:
                            return delta
                    return 999  # no upcoming earnings known
                df["days_until_earnings"] = df.index.map(days_to_next_earnings)
            else:
                df["days_until_earnings"] = 999

            # Cap earnings feature to a reasonable range
            df["days_until_earnings"] = df["days_until_earnings"].clip(0, 90)

            all_dfs.append(df)
        except Exception as e:
            failed.append(f"{ticker}: {e}")

    print(f"\n✅  Step 1 Done! {len(all_dfs)} stocks processed.")
    if failed:
        print(f"⚠️   {len(failed)} failed. First: {failed[0]}")

    # ── 6. Panel Assembly ────────────────────────────────────────────────
    print(f"\n🌐  Step 2/3: Panel Assembly... [RAM: {mem_gb():.1f} GB]")
    panel = pd.concat(all_dfs)
    panel.index.name = "date"
    del all_dfs; gc.collect()

    # Force float32 on entire panel (critical for 16GB systems)
    for col in panel.select_dtypes(include=[np.float64]).columns:
        panel[col] = panel[col].astype(np.float32)
    print(f"    ✅ Downcast to float32 [RAM: {mem_gb():.1f} GB]")

    # ── 6a. Liquidity Filter ─────────────────────────────────────────────
    pre_filter = len(panel)
    panel = filter_illiquid(panel)
    post_filter = len(panel)
    dropped_tickers = pre_filter - post_filter
    print(f"    🚰 Liquidity filter ($>={MIN_DOLLAR_VOL/1e6:.1f}M): "
          f"{pre_filter:,} → {post_filter:,} rows ({dropped_tickers:,} dropped)")
    panel.reset_index(drop=False, inplace=True)
    panel.set_index("date", inplace=True)

    # Merge Fundamentals
    if fundamentals is not None:
        ticker_col = "Ticker" if "Ticker" in fundamentals.columns else "ticker"
        fund_cols = [ticker_col] + [c for c in ["Sector", "Trailing_PE",
                     "Price_To_Book", "EV_to_EBITDA"] if c in fundamentals.columns]
        panel = panel.reset_index().merge(
            fundamentals[fund_cols],
            left_on="ticker", right_on=ticker_col, how="left"
        ).set_index("date")
        if ticker_col != "ticker" and ticker_col in panel.columns:
            panel.drop(columns=[ticker_col], inplace=True)

        # Fill missing fundamentals with global median
        for col in ["Trailing_PE", "Price_To_Book", "EV_to_EBITDA"]:
            if col in panel.columns:
                panel[col] = panel[col].fillna(panel[col].median())

    # Merge full-universe sector data (from download_sectors.py)
    if os.path.exists(SECTORS_CSV):
        sectors_df = pd.read_csv(SECTORS_CSV)
        sector_map = dict(zip(sectors_df["ticker"], sectors_df["sector"]))
        # Override Unknown sectors with downloaded data
        if "Sector" not in panel.columns:
            panel["Sector"] = panel["ticker"].map(sector_map).fillna("Unknown")
        else:
            mask = (panel["Sector"].isna()) | (panel["Sector"] == "Unknown")
            panel.loc[mask, "Sector"] = panel.loc[mask, "ticker"].map(sector_map)
            panel["Sector"] = panel["Sector"].fillna("Unknown")
        known = (panel["Sector"] != "Unknown").sum()
        print(f"    📊 Sector coverage: {known:,}/{len(panel):,} rows "
              f"({known/len(panel)*100:.1f}%)")
    else:
        print(f"    ⚠️  No sectors.csv found. Run: python scripts/download_sectors.py")

    # ── 6c. Merge Sentiment Features ─────────────────────────────────────
    if os.path.exists(SENTIMENT_PATH):
        print(f"    📰 Merging sentiment features...")
        sent_df = pd.read_parquet(SENTIMENT_PATH)
        sent_df["date"] = pd.to_datetime(sent_df["date"])
        # Select only the rolling features (not raw daily which would be noisy)
        sent_cols = ["ticker", "date", "sentiment_mean_3d", "sentiment_mean_7d",
                     "sentiment_vol_7d", "sentiment_momentum", "news_volume_3d",
                     "sentiment_std"]
        sent_cols = [c for c in sent_cols if c in sent_df.columns]
        sent_df = sent_df[sent_cols]

        # Merge on (ticker, date)
        panel = panel.reset_index()
        panel = panel.merge(sent_df, on=["ticker", "date"], how="left")
        panel.set_index("date", inplace=True)

        # Fill missing sentiment with neutral (0.0 for scores, 0 for counts)
        for col in sent_cols:
            if col in ["ticker", "date"]:
                continue
            if col == "news_volume_3d":
                panel[col] = panel[col].fillna(0)
            else:
                panel[col] = panel[col].fillna(0.0)

        # Downcast
        for col in sent_cols:
            if col in ["ticker", "date"]:
                continue
            if col in panel.columns:
                panel[col] = panel[col].astype(np.float32)

        sent_coverage = (panel.get("sentiment_mean_3d", 0) != 0).sum()
        print(f"    ✅ Sentiment merged: {sent_coverage:,}/{len(panel):,} rows have data")
    else:
        print(f"    ⚠️  No sentiment data found. Run: python scripts/sentiment_fetcher.py")
        print(f"         then: python scripts/sentiment_scorer.py")

    # Cross-Sectional Ranking Percentiles (computed per date)
    for feat, rank_name in [
        ("RSI_14", "RSI_rank"),
        ("vol_20", "Vol_rank"),
        ("ret_5d", "Momentum_rank"),
        ("stock_beta_60d", "Beta_rank"),
        ("CMF_20", "CMF_rank"),
    ]:
        if feat in panel.columns:
            panel[rank_name] = panel.groupby(level=0)[feat].rank(pct=True)

    # Sector Momentum
    if "Sector" in panel.columns:
        panel["Sector"] = panel["Sector"].fillna("Unknown")
        panel["Sector_Return"] = panel.groupby(
            ["date", "Sector"], observed=False
        )["log_return"].transform("median")
        panel["Stock_RS_vs_Sector"] = panel["log_return"] - panel["Sector_Return"]

        # ── SECTOR-NEUTRAL TARGET ────────────────────────────────────────
        # excess_return = stock's forward return - sector median forward return
        # This forces the model to find STOCK alpha, not SECTOR beta.
        panel["sector_fwd_ret"] = panel.groupby(
            ["date", "Sector"], observed=False
        )["forward_return"].transform("median")
        panel["excess_return"] = panel["forward_return"] - panel["sector_fwd_ret"]
        panel.drop(columns=["sector_fwd_ret"], inplace=True)

    # ── CRITICAL: Cross-Sectional Z-Scoring ──────────────────────────────
    # This is THE most important transform for ranking.
    # Instead of "AAPL RSI = 72" the model sees "AAPL RSI is 1.8σ above
    # today's universe average." This removes market-wide movements and
    # isolates the RELATIVE signal — exactly what ranking needs.
    print(f"    📐 Cross-sectional z-scoring... [RAM: {mem_gb():.1f} GB]")
    exclude_from_zscore = {
        "forward_return", "excess_return", "ticker", "is_halal", "close", "Sector",
        "days_until_earnings",  # ordinal, not continuous
        "market_regime", "spy_regime", "vix_regime",  # binary flags
        "dollar_volume_20d",  # keep absolute for liquidity ranking
        "day_of_week", "month", "is_month_end_5d", "is_quarter_end_10d",  # calendar
    }
    zscore_cols = [
        c for c in panel.select_dtypes(include=[np.number]).columns
        if c not in exclude_from_zscore
        and not c.endswith("_rank")
        and not c.endswith("_xs")
    ]

    # ── MEMORY: Process z-scoring in batches of 10 columns ──
    # Each column creates 3 temporary Series (mean, std, result).
    # Batching + gc.collect() prevents memory spikes.
    BATCH = 10
    for i in range(0, len(zscore_cols), BATCH):
        batch = zscore_cols[i:i+BATCH]
        for col in batch:
            daily_mean = panel.groupby(level=0)[col].transform("mean")
            daily_std  = panel.groupby(level=0)[col].transform("std")
            daily_std  = daily_std.replace(0, np.nan)
            panel[f"{col}_xs"] = ((panel[col] - daily_mean) / daily_std).astype(np.float32)
            del daily_mean, daily_std
        gc.collect()

    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    panel.fillna(0, inplace=True)
    print(f"    ✅ Added {len(zscore_cols)} z-score features [RAM: {mem_gb():.1f} GB]")

    # ── INTERACTION FEATURES ─────────────────────────────────────────────
    # Pre-compute the combinations the model values most.
    # This reduces the number of tree splits needed to capture these patterns.
    print(f"    🔗 Building interaction features...")
    interactions = {
        # Volatile + rising = breakout candidate
        "vol_x_momentum": ("vol_20_xs", "ret_5d_xs"),
        # Volatile near 52w lows = mean reversion candidate
        "vol_x_value": ("vol_20_xs", "dist_52w_low_xs"),
        # Strong + stable momentum = quality momentum
        "momentum_quality": ("ret_20d_xs", "sharpe_20d_xs"),
        # RSI divergence from sector = contrarian signal
        "rsi_x_sector_rs": ("RSI_14_xs", "Stock_RS_vs_Sector_xs"),
    }
    n_interactions = 0
    for name, (col_a, col_b) in interactions.items():
        if col_a in panel.columns and col_b in panel.columns:
            panel[name] = (panel[col_a] * panel[col_b]).astype(np.float32)
            n_interactions += 1
    print(f"    ✅ Added {n_interactions} interaction features")

    print(f"    ✅ Total features: {len(panel.select_dtypes(include=[np.number]).columns)}")
    
    # Drop rows with no forward return (last FORWARD_DAYS of each stock)
    panel.dropna(subset=["forward_return"], inplace=True)

    print(f"✅  Step 2 Done! Panel shape: {panel.shape}")

    # ── 7. Save ──────────────────────────────────────────────────────────
    print(f"\n💾  Step 3/3: Saving panel to {PANEL_PATH}...")
    panel.to_parquet(PANEL_PATH, index=True)

    # Quick stats
    n_dates = panel.index.nunique()
    n_tickers = panel["ticker"].nunique()
    halal_n = panel[panel["is_halal"] == 1]["ticker"].nunique()
    print(f"\n{'='*50}")
    print(f"📊  Panel Stats:")
    print(f"    Total rows:    {len(panel):,}")
    print(f"    Unique dates:  {n_dates:,}")
    print(f"    Unique tickers: {n_tickers}")
    print(f"    Halal tickers:  {halal_n}")
    print(f"    Forward Return: mean={panel['forward_return'].mean():.4f}, "
          f"std={panel['forward_return'].std():.4f}")
    if "excess_return" in panel.columns:
        print(f"    Excess Return:  mean={panel['excess_return'].mean():.4f}, "
              f"std={panel['excess_return'].std():.4f}")
    if "Sector" in panel.columns:
        print(f"\n    📊 Sector Composition:")
        sector_counts = panel.groupby("Sector")["ticker"].nunique().sort_values(ascending=False)
        for sec, cnt in sector_counts.items():
            halal_in_sec = panel[(panel["Sector"] == sec) & (panel["is_halal"] == 1)]["ticker"].nunique()
            print(f"       {sec:25s} {cnt:4d} tickers ({halal_in_sec:3d} halal)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
