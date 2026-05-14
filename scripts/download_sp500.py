"""
download_sp500.py
-----------------
Downloads OHLCV data for ALL S&P 500 stocks.

The model will be trained on the entire US market to learn universal
price physics. The Halal filter is applied only at inference time.

This script:
1. Scrapes the current S&P 500 ticker list from Wikipedia.
2. Downloads 10 years of daily OHLCV from Yahoo Finance.
3. Saves each ticker as a Parquet file in data/historical/.
4. Skips tickers that already exist (safe to re-run).
"""

import os
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm

DATA_DIR = "data/historical"
PERIOD = "10y"


def get_sp500_tickers() -> list:
    """Get S&P 500 tickers. Uses a curated list."""
    # Curated S&P 500 list (top ~500 US large-cap stocks)
    # We fetch from a reliable raw GitHub source
    import requests
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(resp.text))
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(set(tickers))
    except Exception as e:
        print(f"   ⚠️  Could not fetch from GitHub ({e}), using fallback method...")
        # Fallback: use yfinance to get S&P 500 via the index
        # Use a well-known curated list of major stocks
        sp500_core = [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
            "JPM","V","TSLA","UNH","XOM","MA","JNJ","COST","PG","HD",
            "ABBV","MRK","WMT","NFLX","CRM","BAC","CVX","KO","AMD","PEP",
            "TMO","ORCL","ACN","LIN","MCD","ABT","CSCO","ADBE","WFC","QCOM",
            "DHR","GE","TXN","CAT","INTU","AMAT","PM","AXP","ISRG","BKNG",
            "AMGN","MS","VZ","GS","BLK","PFE","T","SBUX","NOW","HON",
            "NEE","LOW","C","SYK","MDLZ","BA","RTX","SCHW","DE","SPGI",
            "ADI","ETN","TJX","BX","PGR","BMY","ADP","VRTX","LRCX","BSX",
            "MMC","GILD","CB","FI","MU","PANW","SHW","KLAC","REGN","DUK",
            "SO","CL","ICE","CME","MCK","SNPS","CDNS","PH","WM","USB",
            "APD","MSI","EOG","ITW","NOC","MO","PNC","GD","PYPL","WELL",
            "TDG","AJG","CMG","HUM","EMR","CARR","COP","CEG","ORLY","SLB",
            "SPG","AON","CTAS","NXPI","ROP","ECL","FTNT","TFC","OKE","APH",
            "AIG","PSX","COF","GM","FCX","MCO","BK","AZO","AFL","PCAR",
            "SRE","NSC","AEP","TRV","PSA","FDX","D","MPC","HLT","KMB",
            "O","MNST","AMP","PAYX","CCI","ALL","JCI","MCHP","ROST","TGT",
            "GIS","KR","DHI","LHX","MSCI","EW","STZ","HSY","YUM","NEM",
            "WMB","TEL","RCL","IR","IQV","CTSH","IDXX","DD","FAST","OXY",
            "EXC","HAL","A","VRSK","MLM","EXR","KHC","DLTR","DG","DXCM",
            "XEL","EFX","ODFL","BIIB","CPRT","GWW","EBAY","IT","FANG","GEHC",
            "CSGP","ANSS","CDW","KEYS","MTD","TRGP","WAT","PPG","ROK","DOW",
            "AWK","WEC","WST","ZBH","ES","LYV","BRO","TSCO","FTV","STT",
            "BR","HUBB","AXON","STE","HPQ","GLW","LDOS","RJF","CINF","DRI",
            "TROW","EQR","SBAC","VICI","AVB","MOH","TYL","PTC","COO","IRM",
            "DECK","ARE","RF","SNA","TER","BALL","NTRS","PKG","HOLX","MAA",
            "ZBRA","WAB","DPZ","BAX","FDS","J","LUV","CF","PODD","SYF",
            "ESS","HST","WRB","CLX","POOL","CHRW","IP","NVR","KIM","LW",
            "JBHT","AES","PNR","VTRS","MAS","BBY","LH","NDSN","UDR","SWKS",
            "TXT","BEN","CRL","UHS","KMX","CPB","HRL","WYNN","RE","PFG",
            "WBA","MHK","DVA","IVZ","NWS","NWSA","BIO","TAP","BWA","SEE",
            "AAL","PARA","VFC","ZION","FMC","CZR","GNRC","HAS","BBWI","AIZ",
            "MTCH","XRAY","WHR","CTLT","RL"
        ]
        return sorted(set(sp500_core))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load existing Halal tickers to know what we already have
    existing = {f.replace(".parquet", "") for f in os.listdir(DATA_DIR) if f.endswith(".parquet")}

    print("📡 Fetching S&P 500 ticker list from Wikipedia...")
    sp500 = get_sp500_tickers()
    print(f"   Found {len(sp500)} S&P 500 tickers")

    new_tickers = [t for t in sp500 if t not in existing]
    print(f"   Already have {len(existing)} tickers on disk")
    print(f"   Need to download {len(new_tickers)} new tickers\n")

    if not new_tickers:
        print("✅ All S&P 500 tickers already downloaded.")
        return

    failed = []
    for ticker in tqdm(new_tickers, desc="Downloading S&P 500"):
        try:
            df = yf.download(ticker, period=PERIOD, interval="1d", progress=False)
            if df.empty or len(df) < 250:
                failed.append(f"{ticker}: insufficient data ({len(df)} rows)")
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            df.columns = [c.lower() for c in df.columns]
            df.to_parquet(os.path.join(DATA_DIR, f"{ticker}.parquet"))

            # Small delay to respect Yahoo rate limits
            time.sleep(0.05)

        except Exception as e:
            failed.append(f"{ticker}: {e}")

    total_files = len([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
    print(f"\n✅ Download complete!")
    print(f"   Total tickers on disk: {total_files}")
    print(f"   New downloads: {len(new_tickers) - len(failed)}")
    if failed:
        print(f"   ⚠️  {len(failed)} failed:")
        for msg in failed[:10]:
            print(f"      {msg}")


if __name__ == "__main__":
    main()
