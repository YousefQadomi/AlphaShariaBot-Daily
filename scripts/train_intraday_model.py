"""
train_intraday_model.py — Intraday LightGBM Training Pipeline
==============================================================
Trains a short-horizon binary classifier on 5-min bar data for
intraday trading. Predicts whether price will rise ≥ X% within
the next 30 min / 1 hour / 2 hours.

Features are designed to match `intraday_features.py` so the live
engine can reuse the same feature computation at inference time.

Usage:
    python scripts/train_intraday_model.py               # default 30 days
    python scripts/train_intraday_model.py --days 60      # 60-day lookback
    python scripts/train_intraday_model.py --skip-download # use cached bars

Output:
    models/intraday_model.txt            — trained LightGBM model
    models/intraday_features.json        — ordered feature names
    models/intraday_model_metrics.json   — evaluation metrics
    models/intraday_feature_importance.csv
    models/intraday_feature_importance.png
"""

import os
import sys
import gc
import json
import time
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
# pyrefly: ignore [missing-import]
import lightgbm as lgb
# pyrefly: ignore [missing-import]
import optuna
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH       = os.path.join(BASE_DIR, ".env")
HALAL_CSV      = os.path.join(BASE_DIR, "data", "halal_stocks.csv")
BARS_DIR       = os.path.join(BASE_DIR, "data", "intraday_bars")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
MODEL_PATH     = os.path.join(MODEL_DIR, "intraday_model.txt")
FEATURES_PATH  = os.path.join(MODEL_DIR, "intraday_features.json")
METRICS_PATH   = os.path.join(MODEL_DIR, "intraday_model_metrics.json")
FI_CSV_PATH    = os.path.join(MODEL_DIR, "intraday_feature_importance.csv")
FI_PLOT_PATH   = os.path.join(MODEL_DIR, "intraday_feature_importance.png")

ET = ZoneInfo("America/New_York")
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Market hours (ET) — 9:30 to 16:00
MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0

# Bars per day: (16:00 - 9:30) = 390 min / 5 = 78 bars
BARS_PER_DAY = 78

# Data filters
DROP_FIRST_N_BARS = 1   # skip first 5 min (1 bar) — too noisy at open
DROP_LAST_N_BARS  = 6   # skip last 30 min (6 bars) — force-close zone

BATCH_SIZE = 200    # symbols per Alpaca API call
RATE_LIMIT_PAUSE = 0.35  # seconds between batch calls (~170 req/min safe)

TRAIN_RATIO = 0.80
N_OPTUNA_TRIALS = 20

# Target thresholds
TARGET_CONFIGS = {
    "target_30m": {"forward_bars": 6,  "min_gain": 0.010},
    "target_1h":  {"forward_bars": 12, "min_gain": 0.015},
    "target_2h":  {"forward_bars": 24, "min_gain": 0.020},
}

# Which target to train on (primary)
PRIMARY_TARGET = "target_30m"

# Feature names — MUST match intraday_features.py for inference compatibility
FEATURE_NAMES = [
    "vwap_deviation",
    "vwap_slope",
    "relative_volume",
    "rsi_14",
    "macd_hist",
    "stoch_k",
    "ema_9_dist",
    "ema_21_dist",
    "atr_pct",
    "bb_position",
    "momentum_6bar",
    "momentum_12bar",
    "momentum_36bar",
    "momentum_accel",
    "bar_range",
    "close_position_in_bar",
    "session_progress",
    "is_opening_30min",
    "is_power_hour",
    "is_midday",
    "orb_position",
    "volume_surge",
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("TrainIntraday")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def mem_gb():
    """Current RSS memory in GB (cross-platform)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except ImportError:
        pass
    # Fallback: Linux /proc
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    # Windows fallback via ctypes
    try:
        import ctypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(pmc), pmc.cb
        )
        return pmc.WorkingSetSize / (1024 ** 3)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA DOWNLOAD — Alpaca Multi-Symbol Bars
# ──────────────────────────────────────────────────────────────────────────────
class AlpacaBarsFetcher:
    """Downloads 5-min bars for multiple symbols using the batch endpoint."""

    def __init__(self, api_key: str, secret_key: str):
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

    def _get(self, url: str, params: dict = None) -> dict:
        """GET with retry on 429 (rate limit)."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                r = requests.get(
                    url, headers=self.headers, params=params, timeout=30
                )
                if r.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    log.warning(f"  Rate limited (429), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    log.warning(f"  Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Max retries exceeded for Alpaca API")

    def fetch_batch_bars(
        self, symbols: list, start: str, end: str
    ) -> dict:
        """
        Fetch 5-min bars for a batch of symbols using the multi-symbol endpoint.
        Returns: {symbol: [list of bar dicts]}
        Handles pagination via next_page_token.
        """
        all_bars = {s: [] for s in symbols}
        symbols_str = ",".join(symbols)
        page_token = None

        while True:
            params = {
                "symbols": symbols_str,
                "timeframe": "5Min",
                "start": start,
                "end": end,
                "limit": "10000",
                "feed": "iex",
                "adjustment": "raw",
            }
            if page_token:
                params["page_token"] = page_token

            data = self._get(f"{ALPACA_DATA_URL}/v2/stocks/bars", params)

            bars_dict = data.get("bars", {})
            for sym, bars in bars_dict.items():
                all_bars[sym].extend(bars)

            page_token = data.get("next_page_token")
            if not page_token:
                break

            time.sleep(RATE_LIMIT_PAUSE)

        return all_bars

    def download_all(self, tickers: list, days: int) -> dict:
        """
        Download bars for all tickers in batches.
        Returns: {ticker: pd.DataFrame}
        """
        end = datetime.now(ET)
        # Add weekends/holidays buffer
        start = end - timedelta(days=int(days * 1.6) + 5)
        start_str = start.strftime("%Y-%m-%dT00:00:00Z")
        end_str = end.strftime("%Y-%m-%dT23:59:59Z")

        log.info(f"📡 Downloading 5-min bars: {start.date()} → {end.date()}")
        log.info(f"   {len(tickers)} tickers in batches of {BATCH_SIZE}")

        results = {}
        batches = [
            tickers[i : i + BATCH_SIZE]
            for i in range(0, len(tickers), BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(
            tqdm(batches, desc="Fetching bar batches")
        ):
            try:
                batch_bars = self.fetch_batch_bars(batch, start_str, end_str)
                for sym, bars in batch_bars.items():
                    if bars:
                        df = pd.DataFrame(bars)
                        df["t"] = pd.to_datetime(df["t"])
                        df = df.rename(
                            columns={
                                "t": "datetime",
                                "o": "open",
                                "h": "high",
                                "l": "low",
                                "c": "close",
                                "v": "volume",
                                "vw": "vwap",
                            }
                        )
                        # Convert UTC → ET
                        if df["datetime"].dt.tz is not None:
                            df["datetime"] = (
                                df["datetime"]
                                .dt.tz_convert(ET)
                                .dt.tz_localize(None)
                            )
                        else:
                            df["datetime"] = df["datetime"].dt.tz_localize(None)

                        df = df.sort_values("datetime").reset_index(drop=True)
                        # Downcast to float32 immediately
                        for col in ["open", "high", "low", "close", "vwap"]:
                            df[col] = df[col].astype(np.float32)
                        df["volume"] = df["volume"].astype(np.int64)
                        results[sym] = df
            except Exception as e:
                log.warning(f"  ⚠️ Batch {batch_idx} failed: {e}. Falling back to individual fetching for this batch...")
                for sym in batch:
                    try:
                        time.sleep(RATE_LIMIT_PAUSE)
                        single_bar_dict = self.fetch_batch_bars([sym], start_str, end_str)
                        bars = single_bar_dict.get(sym, [])
                        if bars:
                            df = pd.DataFrame(bars)
                            df["t"] = pd.to_datetime(df["t"])
                            df = df.rename(
                                columns={
                                    "t": "datetime",
                                    "o": "open",
                                    "h": "high",
                                    "l": "low",
                                    "c": "close",
                                    "v": "volume",
                                    "vw": "vwap",
                                }
                            )
                            # Convert UTC → ET
                            if df["datetime"].dt.tz is not None:
                                df["datetime"] = (
                                    df["datetime"]
                                    .dt.tz_convert(ET)
                                    .dt.tz_localize(None)
                                )
                            else:
                                df["datetime"] = df["datetime"].dt.tz_localize(None)

                            df = df.sort_values("datetime").reset_index(drop=True)
                            # Downcast to float32 immediately
                            for col in ["open", "high", "low", "close", "vwap"]:
                                df[col] = df[col].astype(np.float32)
                            df["volume"] = df["volume"].astype(np.int64)
                            results[sym] = df
                    except Exception as sym_err:
                        log.debug(f"  ❌ Symbol {sym} failed: {sym_err} — skipping this ticker")

            time.sleep(RATE_LIMIT_PAUSE)

        log.info(f"   ✅ Downloaded bars for {len(results)} tickers")
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. MARKET HOURS FILTER
# ──────────────────────────────────────────────────────────────────────────────
def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars during market hours (9:30 AM - 4:00 PM ET).
    Drop first 5 min (1 bar) and last 30 min (6 bars) of each day."""
    if df.empty:
        return df

    df = df.copy()
    dt = df["datetime"]
    hours = dt.dt.hour
    minutes = dt.dt.minute
    time_mins = hours * 60 + minutes

    # Market hours: 9:30 (570 min) to 16:00 (960 min)
    market_open = MARKET_OPEN_H * 60 + MARKET_OPEN_M   # 570
    market_close = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M  # 960

    mask = (time_mins >= market_open) & (time_mins < market_close)
    df = df[mask].reset_index(drop=True)

    if df.empty:
        return df

    # Drop first N bars and last N bars of each day
    df["_date"] = df["datetime"].dt.date
    filtered_dfs = []
    for _, day_df in df.groupby("_date"):
        n = len(day_df)
        if n <= (DROP_FIRST_N_BARS + DROP_LAST_N_BARS):
            continue
        day_df = day_df.iloc[DROP_FIRST_N_BARS : n - DROP_LAST_N_BARS]
        filtered_dfs.append(day_df)

    if not filtered_dfs:
        return pd.DataFrame()

    result = pd.concat(filtered_dfs, ignore_index=True)
    result.drop(columns=["_date"], inplace=True)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING (vectorized for training)
# ──────────────────────────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI computed without pandas_ta for training speed."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def compute_macd_hist(series: pd.Series) -> pd.Series:
    """MACD histogram (12, 26, 9) normalized by price."""
    ema12 = series.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = series.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    hist = macd_line - signal
    return hist / series.replace(0, np.nan)


def compute_stoch_k(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Stochastic %K."""
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    denom = (highest - lowest).replace(0, np.nan)
    return ((close - lowest) / denom) * 100


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, min_periods=period, adjust=False).mean()


def compute_bb_position(close: pd.Series, period: int = 20) -> pd.Series:
    """Bollinger Band position: -1 (at lower) to +1 (at upper)."""
    sma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    denom = (upper - lower).replace(0, np.nan)
    return ((close - lower) / denom) * 2 - 1


def build_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all intraday features for a single ticker's bar DataFrame.
    Returns DataFrame with feature columns + datetime + date.
    Each row = one 5-min bar.
    """
    if df.empty or len(df) < 40:
        return pd.DataFrame()

    df = df.copy()
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"].astype(np.float64)
    vwap = df["vwap"]

    # ── VWAP features ────────────────────────────────────────────────
    df["vwap_deviation"] = ((c - vwap) / vwap.replace(0, np.nan)).astype(
        np.float32
    )
    df["vwap_slope"] = (
        (vwap - vwap.shift(6)) / vwap.shift(6).replace(0, np.nan)
    ).astype(np.float32)

    # ── Relative volume ──────────────────────────────────────────────
    vol_20avg = v.rolling(20, min_periods=1).mean()
    df["relative_volume"] = (v / vol_20avg.replace(0, np.nan)).clip(0, 20).astype(
        np.float32
    )

    # ── Technical indicators ─────────────────────────────────────────
    df["rsi_14"] = compute_rsi(c, 14).astype(np.float32)
    df["macd_hist"] = compute_macd_hist(c).astype(np.float32)
    df["stoch_k"] = compute_stoch_k(h, lo, c, 14).astype(np.float32)

    ema9 = compute_ema(c, 9)
    ema21 = compute_ema(c, 21)
    df["ema_9_dist"] = ((c - ema9) / c.replace(0, np.nan)).astype(np.float32)
    df["ema_21_dist"] = ((c - ema21) / c.replace(0, np.nan)).astype(np.float32)

    atr = compute_atr(h, lo, c, 14)
    df["atr_pct"] = (atr / c.replace(0, np.nan)).astype(np.float32)

    df["bb_position"] = compute_bb_position(c, 20).clip(-2, 2).astype(np.float32)

    # ── Momentum ─────────────────────────────────────────────────────
    df["momentum_6bar"] = c.pct_change(6).astype(np.float32)
    df["momentum_12bar"] = c.pct_change(12).astype(np.float32)
    df["momentum_36bar"] = c.pct_change(36).astype(np.float32)

    m12 = c.pct_change(12)
    df["momentum_accel"] = (m12 - m12.shift(6)).astype(np.float32)

    # ── Bar microstructure ───────────────────────────────────────────
    bar_range_denom = c.replace(0, np.nan)
    df["bar_range"] = ((h - lo) / bar_range_denom).astype(np.float32)

    hl_denom = (h - lo).replace(0, np.nan)
    df["close_position_in_bar"] = ((c - lo) / hl_denom).astype(np.float32)

    # ── Session context ──────────────────────────────────────────────
    dt = df["datetime"]
    mins_since_open = (
        (dt.dt.hour - MARKET_OPEN_H) * 60
        + (dt.dt.minute - MARKET_OPEN_M)
    ).clip(lower=0)
    df["session_progress"] = (mins_since_open / 390.0).astype(np.float32)
    df["is_opening_30min"] = (mins_since_open < 30).astype(np.int8)
    df["is_power_hour"] = (mins_since_open >= 330).astype(np.int8)  # last hour
    df["is_midday"] = (
        (mins_since_open >= 120) & (mins_since_open < 270)
    ).astype(np.int8)

    # ── Opening Range Breakout ───────────────────────────────────────
    df["_date"] = dt.dt.date
    # First 15 min = first 3 bars of each day
    or_highs = {}
    or_lows = {}
    for date_val, day_df in df.groupby("_date"):
        first_3 = day_df.head(3)
        or_highs[date_val] = first_3["high"].max()
        or_lows[date_val] = first_3["low"].min()

    df["_or_high"] = df["_date"].map(or_highs).astype(np.float32)
    df["_or_low"] = df["_date"].map(or_lows).astype(np.float32)
    or_denom = (df["_or_high"] - df["_or_low"]).replace(0, np.nan)
    df["orb_position"] = (
        ((c - df["_or_low"]) / or_denom) * 2 - 1
    ).clip(-2, 2).astype(np.float32)

    # ── Volume surge ─────────────────────────────────────────────────
    vol_6avg = v.rolling(6, min_periods=1).mean()
    vol_78avg = v.rolling(78, min_periods=1).mean()
    df["volume_surge"] = (vol_6avg > (2 * vol_78avg)).astype(np.int8)

    # Cleanup temp columns
    df.drop(columns=["_date", "_or_high", "_or_low"], inplace=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. TARGET VARIABLE COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each bar at time T, compute forward-looking targets.
    target_X = 1 if max(high) in the next N bars >= close * (1 + threshold).

    Important: targets are computed per-day (no overnight leakage).
    """
    if df.empty:
        return df

    df = df.copy()
    for target_name, cfg in TARGET_CONFIGS.items():
        df[target_name] = np.int8(0)

    df["_date"] = df["datetime"].dt.date
    result_dfs = []

    for _, day_df in df.groupby("_date"):
        day_df = day_df.copy()
        close = day_df["close"].values
        high = day_df["high"].values
        n = len(day_df)

        for target_name, cfg in TARGET_CONFIGS.items():
            fwd = cfg["forward_bars"]
            threshold = cfg["min_gain"]
            targets = np.zeros(n, dtype=np.int8)

            for i in range(n):
                end_idx = min(i + 1 + fwd, n)
                if end_idx <= i + 1:
                    continue
                future_highs = high[i + 1 : end_idx]
                max_high = future_highs.max()
                if max_high >= close[i] * (1 + threshold):
                    targets[i] = 1

            day_df[target_name] = targets

        result_dfs.append(day_df)

    result = pd.concat(result_dfs, ignore_index=True)
    result.drop(columns=["_date"], inplace=True)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def top_decile_precision(targets: np.ndarray, probs: np.ndarray) -> float:
    """Precision of the top 10% most confident predictions."""
    n_top = max(1, int(len(probs) * 0.10))
    top_ix = np.argsort(probs)[::-1][:n_top]
    return float(targets[top_ix].mean())


def top_decile_win_rate(targets: np.ndarray, probs: np.ndarray) -> float:
    """Win rate among the top 10% scored bars."""
    n_top = max(1, int(len(probs) * 0.10))
    top_ix = np.argsort(probs)[::-1][:n_top]
    return float((targets[top_ix] == 1).sum() / len(top_ix))


def find_optimal_threshold(
    targets: np.ndarray, probs: np.ndarray
) -> tuple:
    """Find threshold that maximises F1, requiring at least 10% recall."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.linspace(0.10, 0.90, 81):
        preds = (probs >= thresh).astype(int)
        rec = recall_score(targets, preds, zero_division=0)
        if rec < 0.10:
            continue
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train intraday LightGBM model on 5-min bar data"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of trading days to look back (default: 30)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, use cached parquet files in data/intraday_bars/"
    )
    parser.add_argument(
        "--target", type=str, default=PRIMARY_TARGET,
        choices=list(TARGET_CONFIGS.keys()),
        help=f"Which target to train on (default: {PRIMARY_TARGET})"
    )
    parser.add_argument(
        "--trials", type=int, default=N_OPTUNA_TRIALS,
        help=f"Number of Optuna trials (default: {N_OPTUNA_TRIALS})"
    )
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BARS_DIR, exist_ok=True)

    print("=" * 65)
    print("⚡ AlphaShariaBot — Intraday Model Training Pipeline")
    print("=" * 65)
    print(f"   Target:     {args.target}")
    print(f"   Lookback:   {args.days} trading days")
    print(f"   Trials:     {args.trials}")
    print(f"   RAM:        {mem_gb():.1f} GB")
    print()

    # ── Load API keys ─────────────────────────────────────────────────────
    load_dotenv(ENV_PATH)
    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")

    if not args.skip_download and (not api_key or not secret):
        log.error("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        log.error("   Use --skip-download to use cached data instead.")
        return

    # ── Load halal universe ───────────────────────────────────────────────
    if not os.path.exists(HALAL_CSV):
        log.error(f"❌ Halal stocks file not found: {HALAL_CSV}")
        return

    halal_df = pd.read_csv(HALAL_CSV)
    tickers = halal_df["ticker"].str.upper().str.strip().tolist()
    # Remove tickers with special characters that Alpaca doesn't handle
    tickers = [t for t in tickers if t.isalpha() and t != "XYZ"]
    log.info(f"☪️  Halal universe: {len(tickers)} tickers")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Download or load cached bars
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("📦 STEP 1: Loading 5-minute bar data")
    print(f"{'─'*55}")

    all_bars = {}

    if args.skip_download:
        log.info("⏩ Skipping download, loading cached parquet files...")
        cached_files = [
            f for f in os.listdir(BARS_DIR)
            if f.endswith(".parquet")
        ] if os.path.exists(BARS_DIR) else []

        if not cached_files:
            log.error(f"❌ No cached files found in {BARS_DIR}")
            log.error("   Run without --skip-download first to fetch data.")
            return

        for fname in tqdm(cached_files, desc="Loading cached bars"):
            ticker = fname.replace(".parquet", "").upper()
            fpath = os.path.join(BARS_DIR, fname)
            try:
                df = pd.read_parquet(fpath)
                if not df.empty:
                    all_bars[ticker] = df
            except Exception as e:
                log.warning(f"  Failed to load {fname}: {e}")

        log.info(f"   ✅ Loaded {len(all_bars)} tickers from cache")
    else:
        fetcher = AlpacaBarsFetcher(api_key, secret)
        all_bars = fetcher.download_all(tickers, args.days)

        # Cache to parquet
        log.info(f"💾 Caching bars to {BARS_DIR}...")
        for ticker, df in tqdm(all_bars.items(), desc="Saving parquet"):
            fpath = os.path.join(BARS_DIR, f"{ticker}.parquet")
            df.to_parquet(fpath, index=False, engine="pyarrow")

    if not all_bars:
        log.error("❌ No bar data available. Exiting.")
        return

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Filter market hours + compute features + targets
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("🔧 STEP 2: Feature Engineering & Target Computation")
    print(f"{'─'*55}")

    feature_dfs = []
    skipped = 0
    target_name = args.target

    for ticker in tqdm(list(all_bars.keys()), desc="Computing features"):
        df = all_bars[ticker]

        # Filter to market hours
        df = filter_market_hours(df)
        if df.empty or len(df) < 80:
            skipped += 1
            continue

        # Compute features
        df = build_features_for_ticker(df)
        if df.empty:
            skipped += 1
            continue

        # Compute targets
        df = compute_targets(df)

        # Add ticker column
        df["ticker"] = ticker

        # Keep only rows with valid features (drop warmup period NaNs)
        valid_mask = df[FEATURE_NAMES].notna().all(axis=1)
        df = df[valid_mask]

        if len(df) < 20:
            skipped += 1
            continue

        feature_dfs.append(df)

    # Free raw bars
    del all_bars
    gc.collect()

    if not feature_dfs:
        log.error("❌ No valid feature data produced. Check bar data quality.")
        return

    panel = pd.concat(feature_dfs, ignore_index=True)
    del feature_dfs
    gc.collect()

    # Replace inf/nan
    panel[FEATURE_NAMES] = panel[FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan
    )
    panel[FEATURE_NAMES] = panel[FEATURE_NAMES].fillna(0)

    # Ensure float32
    for col in FEATURE_NAMES:
        panel[col] = panel[col].astype(np.float32)

    log.info(f"\n📊 Panel Statistics:")
    log.info(f"   Total bars:   {len(panel):,}")
    log.info(f"   Tickers:      {panel['ticker'].nunique()}")
    log.info(f"   Skipped:      {skipped}")
    log.info(f"   Date range:   {panel['datetime'].min()} → {panel['datetime'].max()}")
    log.info(f"   Features:     {len(FEATURE_NAMES)}")

    # Target distribution
    for tgt in TARGET_CONFIGS:
        if tgt in panel.columns:
            pct = panel[tgt].mean() * 100
            log.info(f"   {tgt} positive rate: {pct:.1f}%")

    log.info(f"   RAM: {mem_gb():.1f} GB")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Chronological Train/Test Split (time-based, across all stocks)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("✂️  STEP 3: Chronological Train/Test Split")
    print(f"{'─'*55}")

    panel["_date"] = panel["datetime"].dt.date
    unique_dates = sorted(panel["_date"].unique())
    n_dates = len(unique_dates)
    split_idx = int(n_dates * TRAIN_RATIO)

    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    train_mask = panel["_date"].isin(train_dates)
    test_mask = panel["_date"].isin(test_dates)

    X_train = panel.loc[train_mask, FEATURE_NAMES].values
    y_train = panel.loc[train_mask, target_name].values.astype(np.float32)
    X_test = panel.loc[test_mask, FEATURE_NAMES].values
    y_test = panel.loc[test_mask, target_name].values.astype(np.float32)

    del panel
    gc.collect()

    log.info(f"   Train dates:  {len(train_dates)} days")
    log.info(f"   Test dates:   {len(test_dates)} days")
    log.info(f"   Train rows:   {len(X_train):,}")
    log.info(f"   Test rows:    {len(X_test):,}")
    log.info(f"   Train pos %:  {y_train.mean()*100:.1f}%")
    log.info(f"   Test pos %:   {y_test.mean()*100:.1f}%")

    if len(X_train) < 1000:
        log.error("❌ Insufficient training data. Try increasing --days.")
        return

    # ── Class imbalance weighting ─────────────────────────────────────────
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    raw_ratio = neg_count / max(pos_count, 1)
    scale_pos_weight = min(np.sqrt(raw_ratio), 5.0)
    log.info(
        f"   ⚖️  Class ratio: {raw_ratio:.1f}:1 "
        f"(scale_pos_weight: {scale_pos_weight:.2f})"
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Optuna Hyperparameter Tuning
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print(f"🔍 STEP 4: Optuna Hyperparameter Optimization ({args.trials} trials)")
    print(f"{'─'*55}")

    lgb_train = lgb.Dataset(X_train, y_train, feature_name=FEATURE_NAMES)
    lgb_test = lgb.Dataset(
        X_test, y_test, reference=lgb_train, feature_name=FEATURE_NAMES
    )

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42,
            "scale_pos_weight": scale_pos_weight,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 200, 2000
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.4, 0.8
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.6, 0.95
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-3, 10.0, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-3, 10.0, log=True
            ),
        }

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=1500,
            valid_sets=[lgb_test],
            callbacks=callbacks,
        )

        probs = gbm.predict(X_test)
        auc = roc_auc_score(y_test, probs)
        top10 = top_decile_precision(y_test, probs)

        # Dual objective: geometric mean of AUC and Top-10% Precision
        score = np.sqrt(auc * top10) if top10 > 0 else auc * 0.5
        trial.set_user_attr("auc", round(auc, 4))
        trial.set_user_attr("top10_prec", round(top10, 4))
        trial.set_user_attr("n_trees", gbm.num_trees())
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    with tqdm(total=args.trials, desc="Optuna Trials") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix(
                best=f"{study.best_value:.4f}",
                auc=trial.user_attrs.get("auc", "?"),
                prec=trial.user_attrs.get("top10_prec", "?"),
            )
        study.optimize(objective, n_trials=args.trials, callbacks=[callback])

    print(f"\n🏆 Best Trial Score (√AUC×Prec): {study.best_value:.4f}")
    print(f"   AUC:           {study.best_trial.user_attrs.get('auc', '?')}")
    print(f"   Top-10% Prec:  {study.best_trial.user_attrs.get('top10_prec', '?')}")
    print(f"   Trees:         {study.best_trial.user_attrs.get('n_trees', '?')}")

    best_params = study.best_trial.params.copy()
    best_params.update({
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 42,
        "scale_pos_weight": scale_pos_weight,
        "feature_pre_filter": False,
    })

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Train Final Model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("🚂 STEP 5: Training Final Model with Best Parameters")
    print(f"{'─'*55}")

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),
    ]

    final_model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_train, lgb_test],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    final_model.save_model(MODEL_PATH)
    log.info(f"💾 Model saved → {MODEL_PATH} ({final_model.num_trees()} trees)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Evaluation
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("📈 STEP 6: Final Model Evaluation")
    print(f"{'─'*55}")

    probs = final_model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    opt_thresh, best_f1 = find_optimal_threshold(y_test, probs)

    preds = (probs >= opt_thresh).astype(int)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    top10_prec = top_decile_precision(y_test, probs)
    top10_wr = top_decile_win_rate(y_test, probs)

    print(f"\n{'='*50}")
    print(f"  🎯 Target:              {target_name}")
    print(f"  🎯 AUC:                 {auc:.4f}")
    print(f"  🎯 Precision @ Opt:     {prec:.4f}")
    print(f"  🎯 Recall @ Opt:        {rec:.4f}")
    print(f"  🎯 F1-Score @ Opt:      {best_f1:.4f}")
    print(f"  🎯 Top-10% Precision:   {top10_prec:.4f}")
    print(f"  🎯 Top-10% Win Rate:    {top10_wr:.4f}")
    print(f"  📐 Optimal Threshold:   {opt_thresh:.2f}")
    print(f"  🌲 Trees:               {final_model.num_trees()}")
    print(f"{'='*50}\n")

    # ── Save metrics ─────────────────────────────────────────────────────
    metrics = {
        "target": target_name,
        "target_config": TARGET_CONFIGS[target_name],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "auc": round(auc, 4),
        "precision_at_optimal": round(prec, 4),
        "recall_at_optimal": round(rec, 4),
        "f1_at_optimal": round(best_f1, 4),
        "top_10pct_precision": round(top10_prec, 4),
        "top_10pct_win_rate": round(top10_wr, 4),
        "optimal_threshold": round(opt_thresh, 2),
        "num_trees": final_model.num_trees(),
        "num_features": len(FEATURE_NAMES),
        "scale_pos_weight": round(scale_pos_weight, 2),
        "best_params": best_params,
        "trained_at": datetime.now().isoformat(),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"💾 Metrics saved → {METRICS_PATH}")

    # ── Save feature names ───────────────────────────────────────────────
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    log.info(f"💾 Feature list saved → {FEATURES_PATH}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: Feature Importance
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("📊 STEP 7: Feature Importance")
    print(f"{'─'*55}")

    importance = final_model.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance (Gain)": importance,
    }).sort_values(by="Importance (Gain)", ascending=False)

    fi_df.to_csv(FI_CSV_PATH, index=False)
    log.info(f"💾 Feature importance CSV → {FI_CSV_PATH}")

    print("\n🔝 Feature Importance (Top 22):")
    max_imp = fi_df["Importance (Gain)"].max()
    for _, row in fi_df.iterrows():
        bar_len = max(1, int(row["Importance (Gain)"] / max(max_imp, 1) * 30))
        bar = "█" * bar_len
        print(f"   {row['Feature']:28s} {bar} {row['Importance (Gain)']:.0f}")

    # ── Feature importance plot ──────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    plot_df = fi_df.sort_values(by="Importance (Gain)", ascending=True)
    colors = []
    for feat in plot_df["Feature"]:
        if feat in ("vwap_deviation", "vwap_slope"):
            colors.append("#FF6B35")   # VWAP = orange
        elif feat.startswith("momentum") or feat == "momentum_accel":
            colors.append("#2196F3")   # Momentum = blue
        elif feat in ("relative_volume", "volume_surge"):
            colors.append("#9C27B0")   # Volume = purple
        elif feat.startswith("is_") or feat == "session_progress":
            colors.append("#FF9800")   # Session = amber
        else:
            colors.append("#4CAF50")   # Technicals = green

    plt.barh(
        plot_df["Feature"],
        plot_df["Importance (Gain)"],
        color=colors,
    )
    plt.xlabel("Gain (Information provided by feature)")
    plt.title(
        f"Intraday Model — Feature Importance ({target_name})\n"
        f"AUC: {auc:.3f} | Top-10% Precision: {top10_prec:.3f}"
    )
    plt.tight_layout()
    plt.savefig(FI_PLOT_PATH, dpi=150, bbox_inches="tight")
    log.info(f"📊 Feature importance plot → {FI_PLOT_PATH}")

    # ══════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("✅ INTRADAY MODEL TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"   Model:     {MODEL_PATH}")
    print(f"   Features:  {FEATURES_PATH}")
    print(f"   Metrics:   {METRICS_PATH}")
    print(f"   FI CSV:    {FI_CSV_PATH}")
    print(f"   FI Plot:   {FI_PLOT_PATH}")
    print(f"   RAM:       {mem_gb():.1f} GB")
    print()


if __name__ == "__main__":
    main()
