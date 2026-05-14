import os
import gc
import time
import requests
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import concurrent.futures

OUTPUT_PATH = "data/halal_stocks.csv"
DEBT_TO_MC_LIMIT = 0.33
INTEREST_REVENUE_LIMIT = 0.05

# ----------------- إعدادات الأمان للـ Multithreading -----------------
MAX_WORKERS = 8      # عدد الأسهم التي يتم فحصها في نفس اللحظة (لا تزيدها عن 10 لتجنب الحظر)
DELAY_PER_REQ = 0.1  # استراحة خفيفة جداً لكل مسار
# ---------------------------------------------------------------------

def get_sp500_tickers():
    from io import StringIO
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers, timeout=15)
    r.raise_for_status()
    table = pd.read_html(StringIO(r.text))[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()

def is_halal(ticker):
    """
    تقوم هذه الدالة بفحص السهم. أزلنا أمر الـ print لتجنب تداخل النصوص في الشاشة أثناء الـ Multithreading.
    """
    try:
        # تأخير بسيط لحماية الـ IP
        time.sleep(DELAY_PER_REQ)
        
        stock = yf.Ticker(ticker)
        info = stock.info

        # 1. Debt to Market Cap
        total_debt = info.get("totalDebt")
        market_cap = info.get("marketCap")
        
        if total_debt is None: total_debt = 0
        debt_ratio = (total_debt / market_cap) if market_cap else None

        # 2. Interest Income to Revenue
        revenue = info.get("totalRevenue")
        interest_income = 0
        financials = stock.financials

        if not financials.empty:
            if not revenue and 'Total Revenue' in financials.index:
                rev_val = financials.loc['Total Revenue'].iloc[0]
                if pd.notna(rev_val): revenue = rev_val

            for key in ['Interest Income', 'Interest And Investment Income']:
                if key in financials.index:
                    int_val = financials.loc[key].iloc[0]
                    if pd.notna(int_val):
                        interest_income = int_val
                    break

        interest_ratio = (abs(interest_income) / abs(revenue)) if (revenue and revenue > 0) else None

        # الفلترة (الرفض أو القبول)
        if debt_ratio is None or interest_ratio is None:
            return ticker, False

        if debt_ratio >= DEBT_TO_MC_LIMIT or interest_ratio >= INTEREST_REVENUE_LIMIT:
            return ticker, False

        return ticker, True

    except Exception:
        # في حال حدوث خطأ بالسحب، نتجاهل السهم بصمت لتستمر العملية
        return ticker, False

def main():
    os.makedirs("data", exist_ok=True)
    tickers = get_sp500_tickers()
    halal = []

    print(f"Starting screen of {len(tickers)} tickers using Multithreading ({MAX_WORKERS} workers)...")

    # استخدام ThreadPoolExecutor لتشغيل الفحص بشكل متوازٍ
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # تجهيز المهام (Futures)
        future_to_ticker = {executor.submit(is_halal, ticker): ticker for ticker in tickers}
        
        # تتبع التقدم باستخدام tqdm
        with tqdm(total=len(tickers), desc="Fast Screening") as pbar:
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    # جلب نتيجة الدالة: اسم السهم، وهل هو حلال أم لا
                    ticker_name, passed = future.result()
                    if passed:
                        halal.append(ticker_name)
                except Exception as exc:
                    pass  # تم التعامل مع الأخطاء داخل الدالة
                finally:
                    pbar.update(1)

    # تنظيف الذاكرة بعد الانتهاء
    gc.collect()

    # حفظ النتائج
    df = pd.DataFrame(halal, columns=["ticker"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. {len(halal)} halal stocks saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()