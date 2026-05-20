"""
market_monitor.py — Real-Time Market Event Monitor via WebSocket
=================================================================
Uses Alpaca's WebSocket streaming API to monitor real-time events
without polling. Detects volume spikes, ORB breakouts, and price
momentum in real-time.

This module provides event-driven triggers that complement the
periodic scan loop in alpha_intraday.py. When a significant event
is detected (volume surge, breakout, etc.), it notifies the engine
to run an immediate mini-scan on the relevant tickers.

Free Alpaca Plan Constraints:
  - SIP data available via WebSocket (wss://stream.data.alpaca.markets/v2/sip)
  - Can subscribe to trades and quotes for specific symbols
  - Rate: real-time, no polling needed

Usage:
    from market_monitor import MarketMonitor

    monitor = MarketMonitor(api_key, secret_key)
    monitor.set_watchlist(["AAPL", "MSFT", "NVDA"])
    monitor.start()  # starts background thread

    # Check for events
    events = monitor.get_events()
    for event in events:
        print(f"{event['ticker']}: {event['type']} — {event['details']}")

    monitor.stop()
"""

import os
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

log = logging.getLogger("MarketMonitor")
ET = ZoneInfo("America/New_York")

# WebSocket endpoints
WS_SIP_URL = "wss://stream.data.alpaca.markets/v2/sip"
WS_IEX_URL = "wss://stream.data.alpaca.markets/v2/iex"

# Event types
EVENT_VOLUME_SPIKE = "volume_spike"
EVENT_PRICE_BREAKOUT = "price_breakout"
EVENT_MOMENTUM_SURGE = "momentum_surge"


class MarketMonitor:
    """
    Real-time market event monitor using Alpaca WebSocket streaming.

    Tracks price and volume data for a watchlist of tickers and
    detects significant events (volume spikes, breakouts, momentum).

    Falls back to polling mode if websocket is unavailable (e.g.,
    websocket-client not installed).
    """

    def __init__(self, api_key, secret_key, use_sip=False):
        """
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            use_sip: True for SIP feed, False for IEX (free tier)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.ws_url = WS_SIP_URL if use_sip else WS_IEX_URL

        # Watchlist management
        self.watchlist = set()
        self._lock = threading.Lock()

        # Price/volume tracking per ticker
        self._trade_buffer = defaultdict(list)   # {ticker: [(price, volume, timestamp)]}
        self._volume_baseline = {}               # {ticker: avg_volume_per_min}
        self._opening_range = {}                 # {ticker: (or_high, or_low)}

        # Events queue (thread-safe)
        self._events = []
        self._events_lock = threading.Lock()

        # WebSocket state
        self._ws = None
        self._ws_thread = None
        self._running = False
        self._ws_available = False

        # Check if websocket-client is available
        try:
            import websocket  # noqa: F401
            self._ws_available = True
        except ImportError:
            log.warning("  ⚠️ websocket-client not installed — "
                        "MarketMonitor will use polling mode")
            self._ws_available = False

    def set_watchlist(self, tickers):
        """Update the watchlist of tickers to monitor."""
        with self._lock:
            old = self.watchlist.copy()
            self.watchlist = set(t.upper() for t in tickers)

            # If websocket is running, update subscriptions
            if self._running and self._ws and self._ws_available:
                new_tickers = self.watchlist - old
                removed = old - self.watchlist
                if new_tickers:
                    self._subscribe(list(new_tickers))
                if removed:
                    self._unsubscribe(list(removed))

        log.info(f"  📡 Watchlist updated: {len(self.watchlist)} tickers")

    def set_opening_range(self, ticker, or_high, or_low):
        """Set the opening range for breakout detection."""
        self._opening_range[ticker.upper()] = (or_high, or_low)

    def set_volume_baseline(self, ticker, avg_volume_per_min):
        """Set the baseline volume for spike detection."""
        self._volume_baseline[ticker.upper()] = avg_volume_per_min

    def start(self):
        """Start monitoring in a background thread."""
        if self._running:
            return

        self._running = True

        if self._ws_available:
            self._ws_thread = threading.Thread(
                target=self._ws_loop, daemon=True, name="MarketMonitor-WS"
            )
            self._ws_thread.start()
            log.info("  📡 WebSocket market monitor started")
        else:
            self._ws_thread = threading.Thread(
                target=self._poll_loop, daemon=True, name="MarketMonitor-Poll"
            )
            self._ws_thread.start()
            log.info("  📡 Polling market monitor started (install "
                     "'websocket-client' for real-time streaming)")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        log.info("  📡 Market monitor stopped")

    def get_events(self, clear=True):
        """
        Retrieve pending events and optionally clear the queue.
        Each event is a dict with: ticker, type, details, timestamp
        """
        with self._events_lock:
            events = self._events.copy()
            if clear:
                self._events.clear()
        return events

    def get_event_tickers(self, clear=True):
        """Get just the ticker symbols that have pending events."""
        events = self.get_events(clear)
        return list(set(e["ticker"] for e in events))

    # ═══════════════════════════════════════════════════════════════════
    # WebSocket Mode
    # ═══════════════════════════════════════════════════════════════════
    def _ws_loop(self):
        """WebSocket connection loop with auto-reconnect."""
        import websocket

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                log.error(f"  ❌ WebSocket error: {e}")

            if self._running:
                log.info("  🔄 Reconnecting WebSocket in 5s...")
                time.sleep(5)

    def _on_ws_open(self, ws):
        """Authenticate and subscribe on connection."""
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key,
        }
        ws.send(json.dumps(auth_msg))
        log.info("  📡 WebSocket connected, authenticating...")

        # Subscribe to watchlist after auth
        time.sleep(0.5)
        with self._lock:
            if self.watchlist:
                self._subscribe(list(self.watchlist))

    def _on_ws_message(self, ws, message):
        """Process incoming trade/quote messages."""
        try:
            data = json.loads(message)
            if isinstance(data, list):
                for msg in data:
                    self._process_message(msg)
            elif isinstance(data, dict):
                self._process_message(data)
        except Exception as e:
            log.debug(f"  WebSocket message parse error: {e}")

    def _on_ws_error(self, ws, error):
        log.warning(f"  ⚠️ WebSocket error: {error}")

    def _on_ws_close(self, ws, close_code, close_msg):
        log.info(f"  📡 WebSocket closed (code={close_code})")

    def _subscribe(self, tickers):
        """Subscribe to trades for given tickers."""
        if not self._ws:
            return
        sub_msg = {
            "action": "subscribe",
            "trades": tickers,
        }
        try:
            self._ws.send(json.dumps(sub_msg))
            log.debug(f"  Subscribed to {len(tickers)} tickers")
        except Exception as e:
            log.warning(f"  ⚠️ Subscribe failed: {e}")

    def _unsubscribe(self, tickers):
        """Unsubscribe from trades for given tickers."""
        if not self._ws:
            return
        unsub_msg = {
            "action": "unsubscribe",
            "trades": tickers,
        }
        try:
            self._ws.send(json.dumps(unsub_msg))
        except Exception:
            pass

    def _process_message(self, msg):
        """Process a single trade message and detect events."""
        msg_type = msg.get("T")

        if msg_type == "t":  # Trade message
            ticker = msg.get("S", "")
            price = float(msg.get("p", 0))
            volume = int(msg.get("s", 0))
            ts = msg.get("t", "")

            if not ticker or price <= 0:
                return

            # Buffer the trade
            now = time.time()
            self._trade_buffer[ticker].append((price, volume, now))

            # Prune old trades (keep last 5 minutes)
            cutoff = now - 300
            self._trade_buffer[ticker] = [
                t for t in self._trade_buffer[ticker] if t[2] > cutoff
            ]

            # ── Event Detection ───────────────────────────────────────
            self._check_volume_spike(ticker)
            self._check_breakout(ticker, price)
            self._check_momentum(ticker)

    def _check_volume_spike(self, ticker):
        """Detect if recent volume is 3x+ above baseline."""
        baseline = self._volume_baseline.get(ticker)
        if not baseline or baseline <= 0:
            return

        trades = self._trade_buffer.get(ticker, [])
        # Volume in the last 1 minute
        now = time.time()
        recent = [t for t in trades if now - t[2] < 60]
        recent_vol = sum(t[1] for t in recent)

        if recent_vol > baseline * 3:
            self._emit_event(ticker, EVENT_VOLUME_SPIKE, {
                "recent_volume": recent_vol,
                "baseline": baseline,
                "ratio": round(recent_vol / baseline, 1),
            })

    def _check_breakout(self, ticker, price):
        """Detect ORB breakout."""
        or_range = self._opening_range.get(ticker)
        if not or_range:
            return

        or_high, or_low = or_range
        if price > or_high * 1.002:  # 0.2% above OR high
            self._emit_event(ticker, EVENT_PRICE_BREAKOUT, {
                "price": price,
                "or_high": or_high,
                "direction": "up",
                "pct_above": round((price - or_high) / or_high * 100, 2),
            })
            # Remove to avoid repeat events
            del self._opening_range[ticker]

    def _check_momentum(self, ticker):
        """Detect strong short-term momentum (price up >0.5% in 2 min)."""
        trades = self._trade_buffer.get(ticker, [])
        if len(trades) < 10:
            return

        now = time.time()
        # Price 2 minutes ago vs now
        recent_price = trades[-1][0]
        two_min_ago = [t for t in trades if now - t[2] > 100 and now - t[2] < 140]
        if not two_min_ago:
            return

        old_price = two_min_ago[0][0]
        pct_change = (recent_price - old_price) / old_price

        if pct_change > 0.005:  # +0.5% in 2 minutes
            self._emit_event(ticker, EVENT_MOMENTUM_SURGE, {
                "price": recent_price,
                "old_price": old_price,
                "pct_change": round(pct_change * 100, 2),
                "window_sec": 120,
            })

    def _emit_event(self, ticker, event_type, details):
        """Add an event to the queue (deduplication: max 1 per ticker per type per 5 min)."""
        now = time.time()
        with self._events_lock:
            # Check for duplicate (same ticker + type in last 5 min)
            for e in self._events:
                if (e["ticker"] == ticker and e["type"] == event_type and
                        now - e["_ts"] < 300):
                    return  # skip duplicate

            event = {
                "ticker": ticker,
                "type": event_type,
                "details": details,
                "timestamp": datetime.now(ET).strftime("%H:%M:%S ET"),
                "_ts": now,
            }
            self._events.append(event)
            log.info(f"  🚨 EVENT [{event_type}] {ticker}: {details}")

    # ═══════════════════════════════════════════════════════════════════
    # Polling Fallback Mode (when websocket-client not installed)
    # ═══════════════════════════════════════════════════════════════════
    def _poll_loop(self):
        """
        Fallback: poll latest trades via REST API every 30 seconds.
        Less responsive than WebSocket but works without extra dependencies.
        """
        import requests

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        data_url = "https://data.alpaca.markets"

        while self._running:
            try:
                with self._lock:
                    tickers = list(self.watchlist)

                if not tickers:
                    time.sleep(10)
                    continue

                # Batch fetch latest trades
                symbols_str = ",".join(tickers[:200])
                r = requests.get(
                    f"{data_url}/v2/stocks/trades/latest",
                    headers=headers,
                    params={"symbols": symbols_str, "feed": "iex"},
                    timeout=10,
                )

                if r.status_code == 200:
                    trades = r.json().get("trades", {})
                    now = time.time()
                    for ticker, trade in trades.items():
                        price = float(trade.get("p", 0))
                        volume = int(trade.get("s", 0))
                        if price > 0:
                            self._trade_buffer[ticker].append(
                                (price, volume, now)
                            )
                            self._check_breakout(ticker, price)
                            self._check_momentum(ticker)

                time.sleep(30)  # poll every 30 seconds

            except Exception as e:
                log.debug(f"  Poll error: {e}")
                time.sleep(30)

    # ═══════════════════════════════════════════════════════════════════
    # Status
    # ═══════════════════════════════════════════════════════════════════
    def summary(self):
        """Return a summary string of current monitor state."""
        mode = "WebSocket" if self._ws_available else "Polling"
        n_trades = sum(len(v) for v in self._trade_buffer.values())
        with self._events_lock:
            n_events = len(self._events)
        return (f"📡 Monitor: {mode} | "
                f"Watching: {len(self.watchlist)} | "
                f"Buffered: {n_trades} trades | "
                f"Pending events: {n_events}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret:
        print("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY required")
        sys.exit(1)

    print("=" * 60)
    print("📡 Market Monitor — Real-Time Event Detection")
    print("=" * 60)

    monitor = MarketMonitor(api_key, secret)
    test_watchlist = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    monitor.set_watchlist(test_watchlist)
    monitor.start()

    print(f"Monitoring: {test_watchlist}")
    print("Press Ctrl+C to stop...\n")

    try:
        while True:
            events = monitor.get_events()
            if events:
                for e in events:
                    print(f"  🚨 {e['timestamp']} [{e['type']}] "
                          f"{e['ticker']}: {e['details']}")
            time.sleep(5)
    except KeyboardInterrupt:
        monitor.stop()
        print("\n✅ Monitor stopped")
