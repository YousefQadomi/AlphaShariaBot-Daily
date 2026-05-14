"""
download_sectors.py — Fetch sector/industry for all stocks
===========================================================
Fills the critical data gap: V7 had 92% of stocks as "Unknown" sector,
making sector-neutral ranking meaningless.

Usage:
    pip install yfinance
    python scripts/download_sectors.py
"""

import os
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm

INPUT_DIR   = "data/historical"
OUTPUT_PATH = "data/sectors.csv"

def main():
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".parquet"))
    tickers = [f.replace(".parquet", "") for f in files]
    print(f"📡 Downloading sector info for {len(tickers)} tickers...\n")

    # Load existing if resuming
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
        existing = dict(zip(df["ticker"], df["sector"]))
        print(f"   Resuming: {len(existing)} already cached\n")

    results = []
    for ticker in tqdm(tickers, desc="Fetching"):
        if ticker in existing and existing[ticker] != "Unknown":
            results.append({"ticker": ticker, "sector": existing[ticker]})
            continue
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            results.append({"ticker": ticker, "sector": sector if sector else "Unknown"})
        except Exception:
            results.append({"ticker": ticker, "sector": "Unknown"})
        time.sleep(0.1)  # rate limit

        # Save checkpoint every 100
        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)

    known = df[df["sector"] != "Unknown"]
    print(f"\n✅ Done! {len(known)}/{len(df)} tickers have sector data")
    print(f"   Saved → {OUTPUT_PATH}")
    print(f"\n   Sector distribution:")
    print(df["sector"].value_counts().to_string())

if __name__ == "__main__":
    main()
