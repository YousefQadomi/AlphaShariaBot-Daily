import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time

# --- Settings ---
SAVE_DIR = "data/historical"
TIMEFRAME = "10y"
MAX_TICKERS = 3000 

def get_tickers():
    print("📡 Fetching ticker list...")
    urls = [
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt",
        "https://raw.githubusercontent.com/shilewenuw/get_all_tickers/master/get_all_tickers/tickers.csv"
    ]
    
    for url in urls:
        try:
            all_tickers = pd.read_csv(url, header=None, on_bad_lines='skip')[0].dropna().tolist()
            clean_tickers = [str(t).strip().upper() for t in all_tickers if str(t).strip().isalpha() and 1 <= len(str(t).strip()) <= 5]
            clean_tickers = sorted(list(set(clean_tickers)))
            if len(clean_tickers) > 500:
                print(f"✅ Success! Found {len(clean_tickers)} tickers.")
                return clean_tickers[:MAX_TICKERS]
        except:
            continue
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]

def download_universe():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    tickers = get_tickers()
    newly_downloaded = 0
    already_exists = 0
    failed = 0
    
    print(f"🚀 Processing {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers, desc="Downloading Universe"):
        file_path = os.path.join(SAVE_DIR, f"{ticker}.parquet")
        
        if os.path.exists(file_path):
            already_exists += 1
            continue
            
        try:
            # تمت إزالة show_errors لضمان التوافق مع كل النسخ
            df = yf.download(ticker, period=TIMEFRAME, interval="1d", progress=False)
            
            if df is not None and not df.empty and len(df) > 100:
                df.to_parquet(file_path)
                newly_downloaded += 1
            else:
                failed += 1
            
            # تهدئة الطلبات (Rate Limiting)
            if (newly_downloaded + failed) % 25 == 0:
                time.sleep(1)
        except:
            failed += 1

    print(f"\n--- Final Report ---")
    print(f"✅ Newly Downloaded: {newly_downloaded}")
    print(f"📦 Already Existed: {already_exists}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total Dataset Size: {len(os.listdir(SAVE_DIR))} stocks")

if __name__ == "__main__":
    download_universe()