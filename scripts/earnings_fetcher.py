import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time

DATA_DIR = "data/historical"
OUTPUT_FILE = "data/earnings_dates.csv"

def fetch_earnings():
    if not os.path.exists(DATA_DIR):
        print("❌ Error: historical data directory not found!")
        return

    tickers = [f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
    earnings_data = []

    print(f"📡 Fetching upcoming earnings for {len(tickers)} stocks...")

    for ticker in tqdm(tickers, desc="Scanning"):
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            
            next_date = None
            
            # الحالة 1: البيانات تأتي على شكل قاموس (Dictionary)
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                next_date = cal['Earnings Date'][0]
            
            # الحالة 2: البيانات تأتي على شكل جدول (DataFrame)
            elif isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.index:
                    next_date = cal.loc['Earnings Date'].iloc[0]
                elif 'Value' in cal.columns: # بعض النسخ تضع القيمة في عمود Value
                    next_date = cal.loc['Earnings Date', 'Value']

            if next_date:
                earnings_data.append({'ticker': ticker, 'next_earnings_date': next_date})
                
        except:
            continue
            
        # تهدئة الطلبات قليلاً لتجنب الـ 404
        if len(earnings_data) % 15 == 0:
            time.sleep(0.2)

    if earnings_data:
        df_earnings = pd.DataFrame(earnings_data)
        df_earnings.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ SUCCESS! Found earnings dates for {len(earnings_data)} stocks.")
    else:
        print("\n❌ Still no data. Yahoo might be throttling requests.")

if __name__ == "__main__":
    fetch_earnings()