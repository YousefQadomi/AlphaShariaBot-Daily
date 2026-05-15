"""
realtime_news.py — Real-Time News Poller & Catalyst Detector
==============================================================
Polls Alpaca News API every 15 minutes during market hours.
Maintains a rolling 4-hour window of scored headlines.
Emits catalyst alerts when a stock gets sudden news volume.

Features produced:
  - sentiment_flash_15m:  FinBERT score of last 15 min headlines
  - news_spike_count_1h:  Articles in last hour vs 4h average
  - breaking_urgency:     Trigger-word detection score
  - sector_news_momentum: Sector-wide sentiment direction

Usage:
    from realtime_news import IntradayNewsFetcher
    fetcher = IntradayNewsFetcher(alpaca_key, alpaca_secret)
    fetcher.poll()
    catalyst = fetcher.get_catalyst_score("AAPL")
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

log = logging.getLogger("RealtimeNews")

# ─── Trigger Words ────────────────────────────────────────────────────────
BULLISH_TRIGGERS = {
    "beats", "surges", "soars", "upgraded", "fda approved", "approval",
    "merger", "acquisition", "buyback", "record revenue", "raises guidance",
    "strong earnings", "outperforms", "breakout", "all-time high",
    "partnership", "contract win", "dividend increase",
}
BEARISH_TRIGGERS = {
    "misses", "crashes", "downgrades", "downgraded", "recalls", "sec",
    "investigation", "lawsuit", "bankruptcy", "delisted", "warning",
    "profit warning", "cuts guidance", "lowers outlook", "disappoints",
    "loss widens", "supply shortage", "regulatory", "fraud",
}

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
POLL_INTERVAL = 900  # 15 minutes
ROLLING_WINDOW_SEC = 4 * 3600  # 4 hours


class IntradayNewsFetcher:
    """
    Real-time news poller for intraday catalyst detection.
    Maintains an in-memory rolling window of recent headlines.
    """

    def __init__(self, api_key, secret_key, scorer=None):
        """
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            scorer: Optional FinBERTScorer instance for sentiment scoring.
                    If None, uses trigger-word heuristic only.
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.scorer = scorer
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

        # Rolling window: {ticker: [{"headline", "date", "sentiment", "urgency"}]}
        self.news_buffer = defaultdict(list)
        self.last_poll_time = None

    def _trigger_word_score(self, text):
        """
        Fast heuristic: check headline for trigger words.
        Returns (urgency, direction):
          urgency in [0, 1] — how likely this is breaking/actionable news
          direction in [-1, +1] — bullish or bearish
        """
        text_lower = text.lower()
        bull_hits = sum(1 for w in BULLISH_TRIGGERS if w in text_lower)
        bear_hits = sum(1 for w in BEARISH_TRIGGERS if w in text_lower)

        total_hits = bull_hits + bear_hits
        if total_hits == 0:
            return 0.0, 0.0

        urgency = min(1.0, total_hits * 0.3)
        direction = (bull_hits - bear_hits) / total_hits
        return urgency, direction

    def poll(self, tickers=None):
        """
        Fetch latest news from Alpaca. Call this every 15 minutes.

        Args:
            tickers: Optional list of tickers to filter for.
                     If None, fetches all recent news.
        """
        now = datetime.utcnow()
        since = now - timedelta(minutes=20)  # slight overlap to avoid gaps

        params = {
            "start": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 50,
            "sort": "desc",
        }
        if tickers:
            params["symbols"] = ",".join(tickers[:50])

        try:
            r = requests.get(ALPACA_NEWS_URL, headers=self.headers,
                             params=params, timeout=15)
            if r.status_code == 429:
                log.warning("  ⚠️ News API rate limited, will retry next cycle")
                return
            r.raise_for_status()
            data = r.json()
            news_items = data.get("news", [])
        except Exception as e:
            log.warning(f"  ⚠️ News poll failed: {e}")
            return

        new_count = 0
        for item in news_items:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            created = item.get("created_at", "")
            symbols = item.get("symbols", [])

            if not headline or not symbols:
                continue

            # Score the headline
            full_text = f"{headline}. {summary}"
            urgency, direction = self._trigger_word_score(full_text)

            # Use FinBERT if available and headline seems important
            if self.scorer and (urgency > 0.3 or len(headline) > 20):
                try:
                    scores = self.scorer.score_batch([full_text])
                    sentiment = float(scores[0])
                except Exception:
                    sentiment = direction * 0.5
            else:
                sentiment = direction * 0.5

            for ticker in symbols:
                ticker = ticker.upper()
                self.news_buffer[ticker].append({
                    "headline": headline,
                    "created_at": created,
                    "sentiment": sentiment,
                    "urgency": urgency,
                    "direction": direction,
                    "timestamp": time.time(),
                })
                new_count += 1

        # Prune old entries (older than rolling window)
        cutoff = time.time() - ROLLING_WINDOW_SEC
        for ticker in list(self.news_buffer.keys()):
            self.news_buffer[ticker] = [
                n for n in self.news_buffer[ticker] if n["timestamp"] > cutoff
            ]
            if not self.news_buffer[ticker]:
                del self.news_buffer[ticker]

        self.last_poll_time = now
        if new_count > 0:
            log.info(f"  📰 News poll: {new_count} new articles for "
                     f"{len(self.news_buffer)} tickers")

    def get_catalyst_score(self, ticker):
        """
        Compute a catalyst score for a ticker [0, 1].

        0.0 = no news, business as usual
        0.5 = moderate positive news activity
        1.0 = BREAKING: multiple bullish articles, trigger words detected

        Returns:
            (catalyst_score, details_dict)
        """
        entries = self.news_buffer.get(ticker.upper(), [])
        if not entries:
            return 0.0, {"sentiment_flash": 0.0, "news_count_1h": 0,
                         "breaking_urgency": 0.0}

        now = time.time()
        # Last 15 minutes
        recent_15m = [e for e in entries if now - e["timestamp"] < 900]
        # Last 1 hour
        recent_1h = [e for e in entries if now - e["timestamp"] < 3600]
        # All in buffer (up to 4 hours)
        all_entries = entries

        # Sentiment flash (last 15 min average)
        if recent_15m:
            sentiment_flash = np.mean([e["sentiment"] for e in recent_15m])
        elif recent_1h:
            sentiment_flash = np.mean([e["sentiment"] for e in recent_1h]) * 0.7
        else:
            sentiment_flash = np.mean([e["sentiment"] for e in all_entries]) * 0.3

        # News spike: articles in last hour vs 4h average
        hourly_avg = len(all_entries) / 4.0 if all_entries else 0
        news_spike = len(recent_1h) / max(hourly_avg, 0.5)

        # Breaking urgency: max urgency from recent headlines
        max_urgency = max((e["urgency"] for e in recent_1h), default=0)

        # Composite catalyst score
        score = 0.0
        # Positive sentiment contributes up to 0.4
        score += max(0, sentiment_flash) * 0.4
        # News spike contributes up to 0.3
        score += min(0.3, (news_spike - 1) * 0.15) if news_spike > 1 else 0
        # Breaking urgency contributes up to 0.3
        score += max_urgency * 0.3

        score = max(0, min(1.0, score))

        details = {
            "sentiment_flash": round(float(sentiment_flash), 4),
            "news_count_1h": len(recent_1h),
            "news_count_4h": len(all_entries),
            "breaking_urgency": round(float(max_urgency), 4),
            "news_spike_ratio": round(float(news_spike), 2),
        }

        return round(float(score), 4), details

    def get_sector_momentum(self, sector_tickers):
        """
        Compute aggregate sentiment for a group of tickers (sector).
        Returns a score in [-1, +1].
        """
        sentiments = []
        for ticker in sector_tickers:
            entries = self.news_buffer.get(ticker.upper(), [])
            recent = [e for e in entries if time.time() - e["timestamp"] < 3600]
            for e in recent:
                sentiments.append(e["sentiment"])

        if not sentiments:
            return 0.0
        return round(float(np.mean(sentiments)), 4)

    def summary(self):
        """Return a summary of current news buffer state."""
        total = sum(len(v) for v in self.news_buffer.values())
        tickers_with_news = len(self.news_buffer)
        return (f"📰 News Buffer: {total} articles for {tickers_with_news} tickers | "
                f"Last poll: {self.last_poll_time or 'Never'}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    
    # Load env from .env if present
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Error: ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are required.")
        sys.exit(1)
        
    print("=" * 60)
    print("📰 Real-Time News Poller (Intraday Catalyst Detector)")
    print("=" * 60)
    
    # Simple console logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    fetcher = IntradayNewsFetcher(api_key, secret_key)
    print("⏳ Fetching latest market news from Alpaca...")
    fetcher.poll()
    print("\n✅ " + fetcher.summary())
    
    # Show a few top items if any
    if fetcher.news_buffer:
        print("\n🔥 Top recent news items:")
        count = 0
        for ticker, items in fetcher.news_buffer.items():
            for item in items:
                urgency = item['urgency']
                sentiment = item['sentiment']
                headline = item['headline']
                if count < 5:
                    print(f"   [{ticker:5s}] (Urg:{urgency:.1f} Sent:{sentiment:+.2f}) {headline}")
                    count += 1
    print("=" * 60)
