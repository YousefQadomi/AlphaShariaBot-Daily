"""
macro_downloader.py
-------------------
Downloads SPY (S&P 500 proxy) and VIX (CBOE Volatility Index) historical
data and saves them to data/macro/ as snappy-compressed Parquet files.

These files are the single source of truth for all macro context features
injected in feature_engineering.py.  Run this script ONCE (or whenever you
want to refresh the data) BEFORE re-running feature_engineering.py.

Usage:
    python scripts/macro_downloader.py
"""

import os
import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "data/macro"
PERIOD = "10y"
INTERVAL = "1d"

# SPY uses auto_adjust=True (dividend/split adjusted closes).
# VIX is an index — no adjustments needed.
TICKERS = {
    "SPY": {"auto_adjust": True},
    "^VIX": {"auto_adjust": False},
}

# Map from Yahoo ticker symbol to the filename we want on disk.
FILENAME_MAP = {
    "SPY": "SPY.parquet",
    "^VIX": "VIX.parquet",
}


# ---------------------------------------------------------------------------
# Core download
# ---------------------------------------------------------------------------
def download_macro(symbol: str, auto_adjust: bool) -> pd.DataFrame:
    """Download a single macro ticker and return a clean daily DataFrame."""
    df = yf.download(
        symbol,
        period=PERIOD,
        interval=INTERVAL,
        progress=False,
        auto_adjust=auto_adjust,
    )
    if df.empty:
        raise ValueError(f"yfinance returned an empty DataFrame for {symbol}")

    # Flatten MultiIndex columns produced by newer yfinance versions.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalise column names to lowercase.
    df.columns = [c.lower() for c in df.columns]

    # Ensure the index is a proper DatetimeIndex with no timezone.
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    # Drop any completely empty rows (can appear at boundaries).
    df.dropna(how="all", inplace=True)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for symbol, opts in TICKERS.items():
        fname = FILENAME_MAP[symbol]
        out_path = os.path.join(OUTPUT_DIR, fname)
        print(f"⏳  Downloading {symbol} ...", end=" ", flush=True)
        try:
            df = download_macro(symbol, auto_adjust=opts["auto_adjust"])
            df.to_parquet(out_path, engine="pyarrow", compression="snappy")
            print(f"✅  {len(df)} rows → {out_path}  [{df.index[0].date()} → {df.index[-1].date()}]")
        except Exception as exc:
            print(f"❌  FAILED — {exc}")

    print("\n✔  Macro download complete.")


if __name__ == "__main__":
    main()
