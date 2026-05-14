"""
data_expansion.py
------------------
Master Plan Step 1: Data Expansion

Pulls orthogonal information from Yahoo Finance for the Halal Universe.
Specifically, it grabs Sector, Industry, and Key Fundamental Ratios
(P/E, Price-to-Book, EV/EBITDA, Beta).

This data is saved to data/fundamentals.csv and will be used by
feature_engineering.py to provide vital "Value" and "Sector Flow" context
to the LightGBM model, breaking the reliance purely on commoditized
price-geometry.
"""

import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time

DATA_DIR = "data"
UNIVERSE_PATH = os.path.join(DATA_DIR, "halal_stocks.csv")
OUT_PATH = os.path.join(DATA_DIR, "fundamentals.csv")

def get_fundamentals():
    if not os.path.exists(UNIVERSE_PATH):
        raise FileNotFoundError(f"Universe file {UNIVERSE_PATH} not found.")

    df_universe = pd.read_csv(UNIVERSE_PATH)
    if "ticker" not in df_universe.columns:
        # Fallback in case it's actually Ticker
        if "Ticker" in df_universe.columns:
            df_universe.rename(columns={"Ticker": "ticker"}, inplace=True)
        else:
            raise ValueError("halal_stocks.csv must contain a 'ticker' column")

    tickers = df_universe["ticker"].unique()
    
    records = []
    print(f"📡 Fetching fundamental & sector data for {len(tickers)} stocks...")
    
    for ticker in tqdm(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key orthogonal fields safely
            record = {
                "Ticker": ticker,
                "Sector": info.get("sector", "Unknown"),
                "Industry": info.get("industry", "Unknown"),
                "Trailing_PE": info.get("trailingPE", None),
                "Forward_PE": info.get("forwardPE", None),
                "Price_To_Book": info.get("priceToBook", None),
                "EV_to_EBITDA": info.get("enterpriseToEbitda", None),
                "Beta_raw": info.get("beta", None),  # Distinct from our rolling 60d beta
                "Market_Cap": info.get("marketCap", None)
            }
            records.append(record)
            
            # Sleep slightly to avoid yfinance rate limits
            time.sleep(0.1)
            
        except Exception as e:
            # If a ticker fails (e.g. delisted), fill with Unknown/None
            records.append({
                "Ticker": ticker,
                "Sector": "Unknown",
                "Industry": "Unknown",
                "Trailing_PE": None,
                "Forward_PE": None,
                "Price_To_Book": None,
                "EV_to_EBITDA": None,
                "Beta_raw": None,
                "Market_Cap": None
            })
            
    fund_df = pd.DataFrame(records)
    
    # Save
    fund_df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Successfully saved fundamental context to {OUT_PATH}")
    
    # Print a quick summary of sectors found
    sector_counts = fund_df["Sector"].value_counts()
    print("\n📊 Sector Distribution:")
    for sector, count in sector_counts.items():
        print(f"   - {sector}: {count} stocks")

if __name__ == "__main__":
    get_fundamentals()
