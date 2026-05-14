import os
import gc
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

HALAL_CSV = "data/halal_stocks.csv"
OUTPUT_DIR = "data/historical"
FAILED_LOG = "data/failed_tickers.txt"
WORKERS = 15
PERIOD = "10y"
INTERVAL = "1d"


def download_ticker(ticker):
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty dataframe")
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        out_path = os.path.join(OUTPUT_DIR, f"{ticker}.parquet")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        return ticker, None
    except Exception as e:
        return ticker, str(e)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = pd.read_csv(HALAL_CSV)["ticker"].tolist()

    failed = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_ticker, t): t for t in tickers}
        with tqdm(total=len(tickers), desc="Downloading") as pbar:
            for future in as_completed(futures):
                ticker, error = future.result()
                if error:
                    failed.append(ticker)
                pbar.update(1)
                pbar.set_postfix(failed=len(failed))

    if failed:
        with open(FAILED_LOG, "w") as f:
            f.write("\n".join(failed))
        print(f"\n{len(failed)} failed tickers logged to {FAILED_LOG}")

    gc.collect()
    print(f"\nDone. {len(tickers) - len(failed)}/{len(tickers)} tickers saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
