"""
premarket_scanner.py — Pre-Market Gap Scanner
===============================================
Scans the halal universe at 9:25 AM ET to detect pre-market gaps.
Identifies stocks that gapped up/down significantly overnight.

Gap-ups with strong volume = momentum candidates for opening.
Gap-downs = avoid (or contrarian mean-reversion if experienced).

Usage:
    python scripts/premarket_scanner.py              # default 0.5% min gap
    python scripts/premarket_scanner.py --min-gap 1  # 1% min gap
"""

import os
import sys
import logging
import argparse
import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger("PremarketScanner")
ET = ZoneInfo("America/New_York")


class PremarketScanner:
    """
    Pre-market gap scanner.

    Detects stocks that gapped up or down significantly overnight by
    comparing the previous session's close to the current pre-market
    price.  Uses Alpaca batch endpoints internally for speed.
    """

    ALPACA_DATA_URL = "https://data.alpaca.markets"
    MAX_SYMBOLS_PER_REQUEST = 200
    MAX_CONCURRENT_REQUESTS = 5
    MAX_RETRIES = 4
    INITIAL_BACKOFF_SEC = 1.0

    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _request_with_retry(self, method: str, url: str, params: dict = None):
        """HTTP request with exponential back-off on 429 / 5xx."""
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
    # Batch data fetchers (mirrors FastScanner for independence)
    # ──────────────────────────────────────────────────────────────────
    def _fetch_daily_bars_batch(self, tickers: list, lookback_days: int = 5) -> dict:
        """
        Fetch recent daily bars for multiple tickers.

        Returns
        -------
        dict
            ``{ticker: [bar_dict, ...]}`` where each bar dict has
            keys ``t, o, h, l, c, v, vw``.
        """
        now_et = datetime.now(ET)
        start_dt = now_et - timedelta(days=lookback_days)
        start_rfc = start_dt.strftime("%Y-%m-%dT00:00:00-04:00")
        end_rfc = now_et.strftime("%Y-%m-%dT%H:%M:%S-04:00")

        chunks = list(self._chunk_list(tickers, self.MAX_SYMBOLS_PER_REQUEST))
        all_bars: dict = {}

        def _fetch_chunk(chunk):
            symbols_csv = ",".join(chunk)
            url = f"{self.ALPACA_DATA_URL}/v2/stocks/bars"
            params = {
                "symbols": symbols_csv,
                "timeframe": "1Day",
                "start": start_rfc,
                "end": end_rfc,
                "feed": "iex",
                "limit": 10000,
                "adjustment": "raw",
            }
            data = self._request_with_retry("GET", url, params)
            return data.get("bars", {})

        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS) as pool:
            futures = {
                pool.submit(_fetch_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                try:
                    all_bars.update(future.result())
                except Exception as exc:
                    log.error(f"  ❌ Daily bars chunk failed: {exc}")

        return all_bars

    def _fetch_latest_trades_batch(self, tickers: list) -> dict:
        """
        Fetch latest trade price for multiple tickers.

        Returns
        -------
        dict
            ``{ticker: float_price}``
        """
        chunks = list(self._chunk_list(tickers, self.MAX_SYMBOLS_PER_REQUEST))
        all_trades: dict = {}

        def _fetch_chunk(chunk):
            symbols_csv = ",".join(chunk)
            url = f"{self.ALPACA_DATA_URL}/v2/stocks/trades/latest"
            params = {"symbols": symbols_csv, "feed": "iex"}
            data = self._request_with_retry("GET", url, params)
            return data.get("trades", {})

        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS) as pool:
            futures = {
                pool.submit(_fetch_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                try:
                    all_trades.update(future.result())
                except Exception as exc:
                    log.error(f"  ❌ Latest trades chunk failed: {exc}")

        result: dict = {}
        for ticker, raw in all_trades.items():
            try:
                result[ticker] = float(raw["p"])
            except (KeyError, TypeError, ValueError):
                pass
        return result

    def _fetch_avg_volumes(self, daily_bars: dict) -> dict:
        """
        Compute average daily volume from daily bars.

        Parameters
        ----------
        daily_bars : dict
            ``{ticker: [bar_dict, ...]}`` as returned by
            ``_fetch_daily_bars_batch``.

        Returns
        -------
        dict
            ``{ticker: float_avg_volume}``
        """
        result: dict = {}
        for ticker, bars_list in daily_bars.items():
            if not bars_list:
                continue
            volumes = [int(b.get("v", 0)) for b in bars_list]
            avg_vol = np.mean(volumes) if volumes else 0
            if avg_vol > 0:
                result[ticker] = float(avg_vol)
        return result

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def scan_gaps(self, tickers: list, min_gap_pct: float = 0.01) -> list:
        """
        Fetch pre-market data and identify stocks with significant gaps.

        Compares yesterday's closing price to the current (pre-market or
        latest) trade price to detect overnight gaps.

        Parameters
        ----------
        tickers : list of str
            Universe of tickers to scan.
        min_gap_pct : float
            Minimum absolute gap percentage to include (0.01 = 1%).

        Returns
        -------
        list of dict
            Each dict contains::

                {
                    "ticker": "AAPL",
                    "prev_close": 185.50,
                    "premarket_price": 188.20,
                    "gap_pct": 0.0145,       # +1.45%
                    "direction": "up",       # or "down"
                    "volume_ratio": 2.3,     # premarket vol vs avg
                }
        """
        if not tickers:
            return []

        log.info(f"🔍 Pre-market gap scan: {len(tickers)} tickers "
                 f"(min gap: {min_gap_pct * 100:.1f}%)")

        # Step 1: Fetch daily bars (last 5 trading days) — for prev close & avg vol
        log.info("  📡 Fetching daily bars for previous close...")
        daily_bars = self._fetch_daily_bars_batch(tickers, lookback_days=10)

        # Step 2: Fetch latest trade prices (current pre-market or last trade)
        log.info("  📡 Fetching latest trade prices...")
        latest_prices = self._fetch_latest_trades_batch(tickers)

        # Step 3: Compute average volumes for volume ratio
        avg_volumes = self._fetch_avg_volumes(daily_bars)

        # Step 4: Detect gaps
        gaps = []
        for ticker in tickers:
            bars_list = daily_bars.get(ticker)
            current_price = latest_prices.get(ticker)

            if not bars_list or current_price is None:
                continue

            # Previous close = close of the most recent completed daily bar
            try:
                # Sort bars by timestamp descending
                sorted_bars = sorted(bars_list, key=lambda b: b["t"], reverse=True)
                prev_close = float(sorted_bars[0]["c"])
            except (IndexError, KeyError, TypeError):
                continue

            if prev_close <= 0:
                continue

            # Calculate gap
            gap_pct = (current_price - prev_close) / prev_close

            if abs(gap_pct) < min_gap_pct:
                continue

            # Volume ratio (premarket volume isn't directly available from
            # daily bars, so we use the price-change magnitude as a proxy;
            # when intraday bars are available, a real volume ratio can be
            # computed from them).
            avg_vol = avg_volumes.get(ticker, 0)
            # Approximate: use last bar's volume vs average
            last_vol = int(sorted_bars[0].get("v", 0)) if sorted_bars else 0
            volume_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 0.0

            direction = "up" if gap_pct > 0 else "down"

            gaps.append({
                "ticker": ticker,
                "prev_close": round(prev_close, 4),
                "premarket_price": round(current_price, 4),
                "gap_pct": round(gap_pct, 6),
                "direction": direction,
                "volume_ratio": volume_ratio,
            })

        log.info(f"  📊 Found {len(gaps)} stocks with gaps ≥ "
                 f"{min_gap_pct * 100:.1f}%")
        return gaps

    def get_gap_watchlist(
        self,
        tickers: list,
        min_gap_pct: float = 0.005,
    ) -> dict:
        """
        Return categorized gap stocks.

        Parameters
        ----------
        tickers : list of str
            Universe of tickers to scan.
        min_gap_pct : float
            Minimum absolute gap percentage (0.005 = 0.5%).

        Returns
        -------
        dict
            ``{
                "gap_ups":   [list sorted by gap_pct descending],
                "gap_downs": [list sorted by gap_pct ascending],
                "scanned":   int,
                "timestamp": str
            }``

            ``gap_ups`` are momentum candidates for long entries.
            ``gap_downs`` are tickers to avoid (or contrarian plays).
        """
        all_gaps = self.scan_gaps(tickers, min_gap_pct=min_gap_pct)

        gap_ups = sorted(
            [g for g in all_gaps if g["direction"] == "up"],
            key=lambda x: x["gap_pct"],
            reverse=True,
        )
        gap_downs = sorted(
            [g for g in all_gaps if g["direction"] == "down"],
            key=lambda x: x["gap_pct"],
        )

        now_et = datetime.now(ET)
        result = {
            "gap_ups": gap_ups,
            "gap_downs": gap_downs,
            "scanned": len(tickers),
            "timestamp": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        }

        log.info(f"  🟢 Gap-ups:   {len(gap_ups)}")
        if gap_ups:
            top = gap_ups[0]
            log.info(f"     Top: {top['ticker']} +{top['gap_pct'] * 100:.2f}% "
                     f"(${top['prev_close']:.2f} → ${top['premarket_price']:.2f})")
        log.info(f"  🔴 Gap-downs: {len(gap_downs)}")
        if gap_downs:
            worst = gap_downs[0]
            log.info(f"     Worst: {worst['ticker']} {worst['gap_pct'] * 100:.2f}% "
                     f"(${worst['prev_close']:.2f} → ${worst['premarket_price']:.2f})")

        return result


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from dotenv import load_dotenv

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Pre-Market Gap Scanner")
    parser.add_argument(
        "--min-gap", type=float, default=0.5,
        help="Minimum gap percentage (default: 0.5 = 0.5%%)",
    )
    args = parser.parse_args()
    min_gap = args.min_gap / 100.0  # Convert from percentage to decimal

    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        print("❌ Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    scanner = PremarketScanner(api_key, secret)

    # Load halal tickers if available, else use a test set
    HALAL_CSV = os.path.join(BASE_DIR, "data", "halal_stocks.csv")
    if os.path.exists(HALAL_CSV):
        tickers = pd.read_csv(HALAL_CSV)["ticker"].str.upper().tolist()
        print(f"  ☪️ Loaded {len(tickers)} halal tickers")
    else:
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "AMD", "CRM", "NFLX", "ADBE", "INTC", "PYPL", "SQ", "SHOP",
        ]
        print(f"  ℹ️ halal_stocks.csv not found, using {len(tickers)} test tickers")

    print(f"\n{'='*60}")
    print(f"Pre-Market Gap Scanner — {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    print(f"Min gap: {min_gap * 100:.1f}%  |  Tickers: {len(tickers)}")
    print(f"{'='*60}")

    result = scanner.get_gap_watchlist(tickers, min_gap_pct=min_gap)

    print(f"\n{'─'*60}")
    print(f"🟢 GAP-UPS (momentum candidates): {len(result['gap_ups'])}")
    print(f"{'─'*60}")
    for g in result["gap_ups"][:20]:
        print(f"  {g['ticker']:6s}  +{g['gap_pct'] * 100:6.2f}%  "
              f"${g['prev_close']:>8.2f} → ${g['premarket_price']:>8.2f}  "
              f"vol_ratio={g['volume_ratio']:.1f}")

    print(f"\n{'─'*60}")
    print(f"🔴 GAP-DOWNS (avoid list): {len(result['gap_downs'])}")
    print(f"{'─'*60}")
    for g in result["gap_downs"][:20]:
        print(f"  {g['ticker']:6s}  {g['gap_pct'] * 100:6.2f}%  "
              f"${g['prev_close']:>8.2f} → ${g['premarket_price']:>8.2f}  "
              f"vol_ratio={g['volume_ratio']:.1f}")

    print(f"\n{'='*60}")
    print(f"✅ Scan complete @ {result['timestamp']}  |  "
          f"Scanned: {result['scanned']}")
