"""
fast_scanner.py — Batch API Scanner for AlphaShariaBot
======================================================
Replaces sequential per-ticker API calls with batch requests.
Reduces scan time from 3+ minutes to <15 seconds.

Key features:
  - Multi-symbol bar requests (up to 200 per call)
  - Concurrent price/quote fetching via ThreadPoolExecutor
  - Watchlist management (hot list from prior cycles)
  - Rate-limit aware (stays under 200 req/min on free plan)
"""

import requests
import logging
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

log = logging.getLogger("FastScanner")
ET = ZoneInfo("America/New_York")


class FastScanner:
    """Batch API scanner for fast intraday opportunity detection."""

    ALPACA_DATA_URL = "https://data.alpaca.markets"
    MAX_SYMBOLS_PER_REQUEST = 200  # Alpaca limit
    MAX_CONCURRENT_REQUESTS = 5   # Stay under rate limits

    # Retry / back-off settings
    MAX_RETRIES = 4
    INITIAL_BACKOFF_SEC = 1.0

    def __init__(self, api_key: str, secret_key: str):
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        # Watchlist: tickers that showed signals recently (checked first)
        self.watchlist: set = set()
        self.watchlist_scores: dict = {}   # ticker -> last score
        self._last_batch_bars: dict = {}   # cache: ticker -> DataFrame

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _chunk_list(lst, n):
        """Yield successive n-sized chunks from *lst*."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _request_with_retry(self, method: str, url: str, params: dict = None):
        """
        Execute an HTTP request with exponential back-off on 429 / 5xx.
        Returns the parsed JSON response or raises after exhausting retries.
        """
        backoff = self.INITIAL_BACKOFF_SEC
        last_exc = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.request(
                    method, url,
                    headers=self.headers,
                    params=params,
                    timeout=30,
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", backoff))
                    log.warning(
                        f"  ⏳ Rate-limited (429). Retry {attempt}/{self.MAX_RETRIES} "
                        f"in {retry_after:.1f}s"
                    )
                    time.sleep(retry_after)
                    backoff *= 2
                    continue
                if resp.status_code >= 500:
                    log.warning(
                        f"  ⚠️ Server error {resp.status_code}. "
                        f"Retry {attempt}/{self.MAX_RETRIES} in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 400:
                    log.debug(f"  ❌ Client error (400) for {url}: {exc}")
                    raise exc
                last_exc = exc
                log.warning(
                    f"  ⚠️ HTTP error: {exc}. "
                    f"Retry {attempt}/{self.MAX_RETRIES} in {backoff:.1f}s"
                )
                time.sleep(backoff)
                backoff *= 2
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                log.warning(
                    f"  ⚠️ Request error: {exc}. "
                    f"Retry {attempt}/{self.MAX_RETRIES} in {backoff:.1f}s"
                )
                time.sleep(backoff)
                backoff *= 2

        log.error(f"  ❌ Request failed after {self.MAX_RETRIES} retries: {url}")
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Request failed after {self.MAX_RETRIES} retries")

    # ──────────────────────────────────────────────────────────────────
    # Batch bars
    # ──────────────────────────────────────────────────────────────────
    def _fetch_bars_chunk(self, tickers_chunk, timeframe, start_rfc, end_rfc):
        """Fetch bars for a single chunk of tickers (≤200). Returns raw dict."""
        symbols_csv = ",".join(tickers_chunk)
        url = f"{self.ALPACA_DATA_URL}/v2/stocks/bars"
        params = {
            "symbols": symbols_csv,
            "timeframe": timeframe,
            "start": start_rfc,
            "end": end_rfc,
            "feed": "iex",
            "limit": 10000,
            "adjustment": "raw",
        }
        data = self._request_with_retry("GET", url, params)
        return data.get("bars", {})

    def fetch_batch_bars(
        self,
        tickers: list,
        timeframe: str = "5Min",
        lookback_days: int = 5,
    ) -> dict:
        """
        Fetch 5-minute bars for multiple tickers in batch.

        Uses Alpaca's ``/v2/stocks/bars?symbols=AAPL,MSFT,...`` endpoint.
        Splits the ticker list into chunks of MAX_SYMBOLS_PER_REQUEST and
        fetches them concurrently via a ThreadPoolExecutor.

        Returns
        -------
        dict
            ``{ticker: DataFrame}`` where each DataFrame has columns
            ``[datetime, open, high, low, close, volume, vwap]``.
        """
        if not tickers:
            return {}

        now_et = datetime.now(ET)
        start_dt = now_et - timedelta(days=lookback_days)
        start_rfc = start_dt.strftime("%Y-%m-%dT00:00:00-04:00")
        end_rfc = now_et.strftime("%Y-%m-%dT%H:%M:%S-04:00")

        chunks = list(self._chunk_list(tickers, self.MAX_SYMBOLS_PER_REQUEST))
        log.info(
            f"  📡 Fetching bars: {len(tickers)} tickers in "
            f"{len(chunks)} batch(es)  [tf={timeframe}, lookback={lookback_days}d]"
        )

        all_bars: dict = {}

        # Use ThreadPoolExecutor for concurrent chunk fetching
        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS) as pool:
            future_to_idx = {
                pool.submit(
                    self._fetch_bars_chunk, chunk, timeframe, start_rfc, end_rfc
                ): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunk_bars = future.result()
                    all_bars.update(chunk_bars)
                    log.info(
                        f"    ✅ Chunk {idx + 1}/{len(chunks)}: "
                        f"got bars for {len(chunk_bars)} tickers"
                    )
                except Exception as exc:
                    log.error(f"    ❌ Chunk {idx + 1}/{len(chunks)} failed: {exc}. Falling back to individual bar fetching...")
                    chunk = chunks[idx]
                    for sym in chunk:
                        try:
                            time.sleep(0.05)
                            single_bar = self._fetch_bars_chunk([sym], timeframe, start_rfc, end_rfc)
                            all_bars.update(single_bar)
                        except Exception as sym_exc:
                            log.debug(f"      ❌ Bars for {sym} failed: {sym_exc}")

        # Convert raw bar lists into DataFrames
        result: dict = {}
        for ticker, bars_list in all_bars.items():
            if not bars_list:
                continue
            try:
                rows = []
                for bar in bars_list:
                    rows.append(
                        {
                            "datetime": pd.Timestamp(bar["t"]),
                            "open": float(bar["o"]),
                            "high": float(bar["h"]),
                            "low": float(bar["l"]),
                            "close": float(bar["c"]),
                            "volume": int(bar["v"]),
                            "vwap": float(bar.get("vw", 0)),
                        }
                    )
                df = pd.DataFrame(rows)
                df.sort_values("datetime", inplace=True)
                df.reset_index(drop=True, inplace=True)
                result[ticker] = df
            except Exception as exc:
                log.warning(f"  ⚠️ Failed to parse bars for {ticker}: {exc}")

        # Cache for later use
        self._last_batch_bars.update(result)
        log.info(f"  📊 Bars ready for {len(result)}/{len(tickers)} tickers")
        return result

    # ──────────────────────────────────────────────────────────────────
    # Batch quotes
    # ──────────────────────────────────────────────────────────────────
    def _fetch_quotes_chunk(self, tickers_chunk):
        """Fetch latest quotes for a single chunk. Returns raw dict."""
        symbols_csv = ",".join(tickers_chunk)
        url = f"{self.ALPACA_DATA_URL}/v2/stocks/quotes/latest"
        params = {"symbols": symbols_csv, "feed": "iex"}
        data = self._request_with_retry("GET", url, params)
        return data.get("quotes", {})

    def fetch_batch_quotes(self, tickers: list) -> dict:
        """
        Fetch latest bid/ask quotes for multiple tickers in batch.

        Uses ``/v2/stocks/quotes/latest?symbols=AAPL,MSFT,...&feed=sip``

        Returns
        -------
        dict
            ``{ticker: {"bid": float, "ask": float,
                        "bid_size": int, "ask_size": int}}``
        """
        if not tickers:
            return {}

        chunks = list(self._chunk_list(tickers, self.MAX_SYMBOLS_PER_REQUEST))
        log.info(
            f"  📡 Fetching quotes: {len(tickers)} tickers in "
            f"{len(chunks)} batch(es)"
        )

        all_quotes: dict = {}
        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS) as pool:
            future_to_idx = {
                pool.submit(self._fetch_quotes_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunk_quotes = future.result()
                    all_quotes.update(chunk_quotes)
                except Exception as exc:
                    log.error(f"    ❌ Quotes chunk {idx + 1} failed: {exc}. Falling back to individual quote fetching...")
                    chunk = chunks[idx]
                    for sym in chunk:
                        try:
                            time.sleep(0.05)
                            single_quote = self._fetch_quotes_chunk([sym])
                            all_quotes.update(single_quote)
                        except Exception as sym_exc:
                            log.debug(f"      ❌ Quote for {sym} failed: {sym_exc}")

        # Normalise into clean dicts
        result: dict = {}
        for ticker, raw in all_quotes.items():
            try:
                result[ticker] = {
                    "bid": float(raw.get("bp", 0)),
                    "ask": float(raw.get("ap", 0)),
                    "bid_size": int(raw.get("bs", 0)),
                    "ask_size": int(raw.get("as", 0)),
                }
            except (TypeError, ValueError) as exc:
                log.warning(f"  ⚠️ Bad quote data for {ticker}: {exc}")

        log.info(f"  📊 Quotes ready for {len(result)}/{len(tickers)} tickers")
        return result

    # ──────────────────────────────────────────────────────────────────
    # Batch trades (latest price)
    # ──────────────────────────────────────────────────────────────────
    def _fetch_trades_chunk(self, tickers_chunk):
        """Fetch latest trades for a single chunk. Returns raw dict."""
        symbols_csv = ",".join(tickers_chunk)
        url = f"{self.ALPACA_DATA_URL}/v2/stocks/trades/latest"
        params = {"symbols": symbols_csv, "feed": "iex"}
        data = self._request_with_retry("GET", url, params)
        return data.get("trades", {})

    def fetch_batch_trades(self, tickers: list) -> dict:
        """
        Fetch the latest trade price for multiple tickers in batch.

        Uses ``/v2/stocks/trades/latest?symbols=AAPL,MSFT,...&feed=sip``

        Returns
        -------
        dict
            ``{ticker: price}`` where *price* is a float.
        """
        if not tickers:
            return {}

        chunks = list(self._chunk_list(tickers, self.MAX_SYMBOLS_PER_REQUEST))
        log.info(
            f"  📡 Fetching trades: {len(tickers)} tickers in "
            f"{len(chunks)} batch(es)"
        )

        all_trades: dict = {}
        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS) as pool:
            future_to_idx = {
                pool.submit(self._fetch_trades_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunk_trades = future.result()
                    all_trades.update(chunk_trades)
                except Exception as exc:
                    log.error(f"    ❌ Trades chunk {idx + 1} failed: {exc}. Falling back to individual trade fetching...")
                    chunk = chunks[idx]
                    for sym in chunk:
                        try:
                            time.sleep(0.05)
                            single_trade = self._fetch_trades_chunk([sym])
                            all_trades.update(single_trade)
                        except Exception as sym_exc:
                            log.debug(f"      ❌ Trade for {sym} failed: {sym_exc}")

        result: dict = {}
        for ticker, raw in all_trades.items():
            try:
                result[ticker] = float(raw["p"])
            except (KeyError, TypeError, ValueError) as exc:
                log.warning(f"  ⚠️ Bad trade data for {ticker}: {exc}")

        log.info(f"  📊 Prices ready for {len(result)}/{len(tickers)} tickers")
        return result

    # ──────────────────────────────────────────────────────────────────
    # Watchlist management
    # ──────────────────────────────────────────────────────────────────
    WATCHLIST_MAX_SIZE = 40

    def update_watchlist(self, scored_tickers: list):
        """
        Update the hot watchlist based on recent scoring results.

        Parameters
        ----------
        scored_tickers : list of dict
            Each dict must have at least ``{"ticker": str, "score": float}``.
            Typically the output of a scan cycle.

        Keeps the top ``WATCHLIST_MAX_SIZE`` tickers by score across
        the current watchlist and newly scored tickers.
        """
        # Merge new scores into existing scores
        for item in scored_tickers:
            ticker = item["ticker"]
            score = item["score"]
            # Exponential moving average: blend old/new score
            if ticker in self.watchlist_scores:
                self.watchlist_scores[ticker] = (
                    0.3 * self.watchlist_scores[ticker] + 0.7 * score
                )
            else:
                self.watchlist_scores[ticker] = score

        # Sort by score descending, keep top N
        sorted_tickers = sorted(
            self.watchlist_scores.items(), key=lambda x: x[1], reverse=True
        )
        top = sorted_tickers[: self.WATCHLIST_MAX_SIZE]
        self.watchlist = {t for t, _ in top}
        self.watchlist_scores = {t: s for t, s in top}

        log.info(
            f"  🔥 Watchlist updated: {len(self.watchlist)} tickers  "
            f"(top score: {top[0][1]:.1f})" if top else
            "  🔥 Watchlist updated: 0 tickers"
        )

    def get_scan_order(self, all_tickers: list) -> list:
        """
        Return tickers in priority order:
          1. Watchlist tickers (showed signals before) — sorted by score
          2. Remaining tickers — original order preserved

        Parameters
        ----------
        all_tickers : list of str

        Returns
        -------
        list of str
        """
        all_set = set(all_tickers)

        # Watchlist tickers that are still in the universe, sorted by score
        wl_sorted = sorted(
            [t for t in self.watchlist if t in all_set],
            key=lambda t: self.watchlist_scores.get(t, 0),
            reverse=True,
        )
        wl_set = set(wl_sorted)

        # Remaining tickers, preserve original order
        rest = [t for t in all_tickers if t not in wl_set]

        ordered = wl_sorted + rest
        if wl_sorted:
            log.info(
                f"  📋 Scan order: {len(wl_sorted)} watchlist first, "
                f"then {len(rest)} others"
            )
        return ordered


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys
    from dotenv import load_dotenv

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        print("❌ Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    scanner = FastScanner(api_key, secret)

    # Small test batch
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    print(f"\n{'='*55}")
    print(f"Testing FastScanner with {len(test_tickers)} tickers")
    print(f"{'='*55}")

    # 1. Batch bars
    print("\n── Batch Bars ──")
    bars = scanner.fetch_batch_bars(test_tickers, timeframe="5Min", lookback_days=2)
    for ticker, df in bars.items():
        print(f"  {ticker}: {len(df)} bars, "
              f"latest close=${df['close'].iloc[-1]:.2f}" if len(df) > 0 else
              f"  {ticker}: 0 bars")

    # 2. Batch quotes
    print("\n── Batch Quotes ──")
    quotes = scanner.fetch_batch_quotes(test_tickers)
    for ticker, q in quotes.items():
        spread = q["ask"] - q["bid"]
        spread_pct = (spread / q["ask"] * 100) if q["ask"] > 0 else 0
        print(f"  {ticker}: bid=${q['bid']:.2f} ask=${q['ask']:.2f} "
              f"spread={spread_pct:.3f}%")

    # 3. Batch trades
    print("\n── Batch Trades ──")
    prices = scanner.fetch_batch_trades(test_tickers)
    for ticker, price in prices.items():
        print(f"  {ticker}: ${price:.2f}")

    # 4. Watchlist
    print("\n── Watchlist ──")
    mock_scores = [
        {"ticker": "AAPL", "score": 72.5},
        {"ticker": "NVDA", "score": 88.0},
        {"ticker": "TSLA", "score": 61.2},
        {"ticker": "MSFT", "score": 55.0},
    ]
    scanner.update_watchlist(mock_scores)
    ordered = scanner.get_scan_order(test_tickers)
    print(f"  Scan order: {ordered}")

    print(f"\n{'='*55}")
    print("✅ FastScanner self-test complete")
