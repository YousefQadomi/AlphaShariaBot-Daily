"""
sentiment_fetcher.py — Download News Headlines for Sentiment Analysis
======================================================================
Fetches news headlines from two free APIs the project already has keys for:
  1. Alpaca News API (real-time financial news, free with paper account)
  2. Alpha Vantage News Sentiment API (broader coverage, free tier)

The raw headlines are saved to data/sentiment/raw_news.parquet.
Run sentiment_scorer.py next to score them with FinBERT.

Usage:
    python scripts/sentiment_fetcher.py
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH    = os.path.join(BASE_DIR, ".env")
HALAL_CSV   = os.path.join(BASE_DIR, "data", "halal_stocks.csv")
HIST_DIR    = os.path.join(BASE_DIR, "data", "historical")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "sentiment")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "raw_news.parquet")

# ─── Config ───────────────────────────────────────────────────────────────
# How many days back to fetch news (Alpaca free tier: up to 5 years)
LOOKBACK_DAYS     = 365 * 2   # 2 years of news
ALPACA_NEWS_URL   = "https://data.alpaca.markets/v1beta1/news"
AV_NEWS_URL       = "https://www.alphavantage.co/query"
BATCH_SIZE        = 10        # tickers per Alpaca request
DELAY_BETWEEN_REQ = 0.2      # seconds between API calls
AV_CALLS_PER_MIN  = 5        # Alpha Vantage free tier limit


# ═══════════════════════════════════════════════════════════════════════════
# 1. Alpaca News API
# ═══════════════════════════════════════════════════════════════════════════
def fetch_alpaca_news(tickers: list, api_key: str, secret_key: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch news from Alpaca's free News API.
    Supports multi-ticker batch queries and pagination.
    """
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    all_articles = []

    # Process tickers in batches
    for i in tqdm(range(0, len(tickers), BATCH_SIZE), desc="Alpaca News"):
        batch = tickers[i:i + BATCH_SIZE]
        symbols = ",".join(batch)

        page_token = None
        pages_fetched = 0
        max_pages = 10  # cap per batch to avoid rate limits

        while pages_fetched < max_pages:
            params = {
                "symbols": symbols,
                "start": f"{start_date}T00:00:00Z",
                "end": f"{end_date}T23:59:59Z",
                "limit": 50,
                "sort": "desc",
            }
            if page_token:
                params["page_token"] = page_token

            try:
                r = requests.get(ALPACA_NEWS_URL, headers=headers,
                                 params=params, timeout=15)
                if r.status_code == 429:
                    time.sleep(5)
                    continue
                r.raise_for_status()
                data = r.json()

                news_items = data.get("news", [])
                if not news_items:
                    break

                for item in news_items:
                    # Each article may relate to multiple tickers
                    article_tickers = [s for s in item.get("symbols", [])
                                       if s in set(tickers)]
                    if not article_tickers:
                        # If no specific tickers matched, skip
                        continue

                    for t in article_tickers:
                        all_articles.append({
                            "ticker": t,
                            "date": item.get("created_at", "")[:10],
                            "headline": item.get("headline", ""),
                            "summary": item.get("summary", ""),
                            "source": item.get("source", "alpaca"),
                            "url": item.get("url", ""),
                            "api": "alpaca",
                        })

                page_token = data.get("next_page_token")
                if not page_token:
                    break
                pages_fetched += 1

            except requests.exceptions.RequestException as e:
                print(f"  ⚠️  Alpaca API error for batch {batch[:3]}: {e}")
                break

            time.sleep(DELAY_BETWEEN_REQ)

    return pd.DataFrame(all_articles)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Alpha Vantage News API
# ═══════════════════════════════════════════════════════════════════════════
def fetch_alpha_vantage_news(tickers: list, av_key: str) -> pd.DataFrame:
    """
    Fetch news from Alpha Vantage News Sentiment endpoint.
    Free tier: 25 requests/day, so we focus on top tickers.
    """
    all_articles = []
    calls_made = 0

    # Only fetch for a subset due to free tier limits
    max_tickers = min(len(tickers), 20)  # 20 tickers max for free tier
    subset = tickers[:max_tickers]

    for ticker in tqdm(subset, desc="Alpha Vantage News"):
        if calls_made >= 24:  # leave 1 call buffer
            print("  ⚠️  Alpha Vantage daily limit approaching, stopping.")
            break

        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": 200,
                "apikey": av_key,
            }
            r = requests.get(AV_NEWS_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            feed = data.get("feed", [])
            for item in feed:
                pub_date = item.get("time_published", "")[:8]
                if len(pub_date) == 8:
                    pub_date = f"{pub_date[:4]}-{pub_date[4:6]}-{pub_date[6:8]}"

                all_articles.append({
                    "ticker": ticker,
                    "date": pub_date,
                    "headline": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", "alphavantage"),
                    "url": item.get("url", ""),
                    "api": "alphavantage",
                })

            calls_made += 1

        except Exception as e:
            print(f"  ⚠️  Alpha Vantage error for {ticker}: {e}")

        time.sleep(60 / AV_CALLS_PER_MIN)  # respect rate limit

    return pd.DataFrame(all_articles)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_dotenv(ENV_PATH)

    # Load API keys
    alpaca_key    = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    av_key        = os.getenv("ALPHA_VANTAGE_KEY")

    # Get all tickers we have historical data for
    hist_tickers = sorted(
        f.replace(".parquet", "")
        for f in os.listdir(HIST_DIR)
        if f.endswith(".parquet")
    )
    print(f"📰 Sentiment Fetcher — {len(hist_tickers)} tickers\n")

    # Date range
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    print(f"   Date range: {start_date} → {end_date}")

    all_dfs = []

    # ── Alpaca News ───────────────────────────────────────────────────
    if alpaca_key and alpaca_secret:
        print(f"\n📡 Fetching from Alpaca News API...")
        df_alpaca = fetch_alpaca_news(
            hist_tickers, alpaca_key, alpaca_secret, start_date, end_date
        )
        if not df_alpaca.empty:
            print(f"   ✅ Alpaca: {len(df_alpaca):,} articles")
            all_dfs.append(df_alpaca)
        else:
            print("   ⚠️  Alpaca returned 0 articles")
    else:
        print("⚠️  No Alpaca keys in .env — skipping Alpaca News")

    # ── Alpha Vantage News ────────────────────────────────────────────
    if av_key:
        print(f"\n📡 Fetching from Alpha Vantage News API...")
        df_av = fetch_alpha_vantage_news(hist_tickers, av_key)
        if not df_av.empty:
            print(f"   ✅ Alpha Vantage: {len(df_av):,} articles")
            all_dfs.append(df_av)
        else:
            print("   ⚠️  Alpha Vantage returned 0 articles")
    else:
        print("⚠️  No ALPHA_VANTAGE_KEY in .env — skipping Alpha Vantage")

    # ── Combine & Save ────────────────────────────────────────────────
    if not all_dfs:
        print("\n❌ No news data fetched from any source. Check API keys.")
        return

    news = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate by (ticker, date, headline)
    news.drop_duplicates(subset=["ticker", "date", "headline"], inplace=True)

    # Clean
    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    news.dropna(subset=["date", "headline"], inplace=True)
    news = news[news["headline"].str.len() > 10]  # drop empty/garbage headlines

    news.sort_values(["ticker", "date"], inplace=True)
    news.reset_index(drop=True, inplace=True)

    news.to_parquet(OUTPUT_PATH, index=False)

    # Stats
    n_tickers = news["ticker"].nunique()
    n_days    = news["date"].nunique()
    print(f"\n{'='*50}")
    print(f"📊 News Data Summary:")
    print(f"   Total articles:    {len(news):,}")
    print(f"   Unique tickers:    {n_tickers}")
    print(f"   Date range:        {news['date'].min().date()} → {news['date'].max().date()}")
    print(f"   Unique dates:      {n_days}")
    print(f"   Avg articles/day:  {len(news)/max(n_days,1):.1f}")
    print(f"   Saved → {OUTPUT_PATH}")
    print(f"{'='*50}")
    print(f"\n✅ Next step: python scripts/sentiment_scorer.py")


if __name__ == "__main__":
    main()
