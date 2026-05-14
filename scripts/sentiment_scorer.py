"""
sentiment_scorer.py — FinBERT Sentiment Scoring Pipeline
=========================================================
Scores news headlines with ProsusAI/finbert, aggregates to daily
per-ticker sentiment features for the ranking model.

Output: data/sentiment/daily_sentiment.parquet

Usage:
    python scripts/sentiment_scorer.py
"""

import os, gc, warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_NEWS_PATH = os.path.join(BASE_DIR, "data", "sentiment", "raw_news.parquet")
OUTPUT_PATH   = os.path.join(BASE_DIR, "data", "sentiment", "daily_sentiment.parquet")

MODEL_NAME     = "ProsusAI/finbert"
BATCH_SIZE     = 64
MAX_SEQ_LENGTH = 128
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


class FinBERTScorer:
    """Wraps ProsusAI/finbert. Outputs score in [-1, +1]."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"🧠 Loading FinBERT on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()
        print(f"   ✅ FinBERT loaded")

    @torch.no_grad()
    def score_batch(self, texts: list) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        # positive*1 + negative*(-1) + neutral*0
        scores = probs[:, 0] * 1.0 + probs[:, 1] * (-1.0)
        return scores.astype(np.float32)


def score_all_headlines(news: pd.DataFrame) -> pd.DataFrame:
    scorer = FinBERTScorer()
    texts = (news["headline"].fillna("") + ". " + news["summary"].fillna("")).tolist()
    all_scores = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT Scoring"):
        batch = texts[i:i + BATCH_SIZE]
        scores = scorer.score_batch(batch)
        all_scores.extend(scores)
        if i % (BATCH_SIZE * 50) == 0:
            gc.collect()
    news["sentiment"] = all_scores
    del scorer; gc.collect()
    return news


def aggregate_daily(news: pd.DataFrame) -> pd.DataFrame:
    daily = news.groupby(["ticker", "date"]).agg(
        sentiment_mean=("sentiment", "mean"),
        sentiment_std=("sentiment", "std"),
        news_count=("sentiment", "count"),
    ).reset_index()
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    daily.sort_values(["ticker", "date"], inplace=True)

    result_dfs = []
    for ticker, group in daily.groupby("ticker"):
        g = group.copy().set_index("date")
        if len(g) < 2:
            continue
        idx = pd.date_range(g.index.min(), g.index.max(), freq="B")
        g = g.reindex(idx)
        g["sentiment_mean"] = g["sentiment_mean"].ffill()
        g["sentiment_std"]  = g["sentiment_std"].ffill().fillna(0)
        g["news_count"]     = g["news_count"].fillna(0)
        g["sentiment_mean_3d"]  = g["sentiment_mean"].rolling(3, min_periods=1).mean()
        g["sentiment_mean_7d"]  = g["sentiment_mean"].rolling(7, min_periods=1).mean()
        g["sentiment_vol_7d"]   = g["sentiment_mean"].rolling(7, min_periods=1).std().fillna(0)
        g["sentiment_momentum"] = g["sentiment_mean_3d"] - g["sentiment_mean_7d"]
        g["news_volume_3d"]     = g["news_count"].rolling(3, min_periods=1).sum()
        g["ticker"] = ticker
        g.index.name = "date"
        g.reset_index(inplace=True)
        result_dfs.append(g)

    if not result_dfs:
        return pd.DataFrame()
    result = pd.concat(result_dfs, ignore_index=True)
    cols = ["ticker", "date", "sentiment_mean", "sentiment_std", "news_count",
            "sentiment_mean_3d", "sentiment_mean_7d", "sentiment_vol_7d",
            "sentiment_momentum", "news_volume_3d"]
    result = result[cols]
    for col in result.select_dtypes(include=[np.float64]).columns:
        result[col] = result[col].astype(np.float32)
    return result


def main():
    print("🧠 AlphaShariaBot — FinBERT Sentiment Scorer\n")
    if not os.path.exists(RAW_NEWS_PATH):
        print(f"❌ Raw news not found at {RAW_NEWS_PATH}")
        print("   Run 'python scripts/sentiment_fetcher.py' first.")
        return

    news = pd.read_parquet(RAW_NEWS_PATH)
    news["date"] = pd.to_datetime(news["date"])
    print(f"   {len(news):,} articles for {news['ticker'].nunique()} tickers")

    print(f"\n📊 Scoring {len(news):,} headlines with FinBERT on {DEVICE}...")
    news = score_all_headlines(news)

    print(f"\n📐 Aggregating to daily per-ticker features...")
    daily = aggregate_daily(news)
    if daily.empty:
        print("❌ No daily sentiment data produced.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    daily.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"📊 Daily Sentiment Summary:")
    print(f"   Rows: {len(daily):,} | Tickers: {daily['ticker'].nunique()}")
    print(f"   Range: {daily['date'].min().date()} → {daily['date'].max().date()}")
    print(f"   Saved → {OUTPUT_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
