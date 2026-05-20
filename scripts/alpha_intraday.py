"""
alpha_intraday.py — Intraday Day Trading Engine V2 for AlphaShariaBot
======================================================================
COMPLETE REWRITE: Integrates batch scanning, ATR-based risk management,
ML model support, smart order execution, and adaptive scan frequency.

Key V2 improvements:
  - Batch API scanning via FastScanner (333 tickers in <15s vs 3+ min)
  - ATR-based dynamic stop-losses (adapts to each stock's volatility)
  - Scaling exits: partial at +0.8%, trail remainder to +2.0%
  - Time stops: close stale positions after 90 minutes
  - Smart order execution with fill verification
  - Adaptive scan frequency (60s opening, 10min midday, 2min power hour)
  - Pre-market gap scanning for momentum candidates
  - Position reconciliation on startup (wallet vs Alpaca)
  - Small-account optimization ($100 capital with $15 min positions)
  - News-driven event triggers for fast entries

Sharia Compliance:
  - Long-only (no shorting) — HARDCODED
  - No margin trading
  - Halal-screened universe only
  - Fractional shares = real ownership

Usage:
    python scripts/alpha_intraday.py              # single scan cycle
    python scripts/alpha_intraday.py --loop        # continuous loop
    python scripts/alpha_intraday.py --force-close # liquidate all
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import gc
import json
import time
import logging
import requests
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_manager import IntradayRiskManager
from intraday_features import IntradayFeatureEngine
from realtime_news import IntradayNewsFetcher
from fast_scanner import FastScanner
from premarket_scanner import PremarketScanner
from market_monitor import MarketMonitor

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALLET_PATH    = os.path.join(BASE_DIR, "data", "live", "intraday_wallet.json")
HALAL_CSV      = os.path.join(BASE_DIR, "data", "halal_stocks.csv")
FUND_CSV       = os.path.join(BASE_DIR, "data", "fundamentals.csv")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
ENV_PATH       = os.path.join(BASE_DIR, ".env")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "intraday_model.txt")
FEATURES_PATH  = os.path.join(BASE_DIR, "models", "intraday_features.json")
METRICS_PATH   = os.path.join(BASE_DIR, "models", "intraday_model_metrics.json")
ET             = ZoneInfo("America/New_York")

# ─── Strategy Constants ───────────────────────────────────────────────────
INITIAL_BALANCE     = 1000.0
MIN_ENTRY_SCORE     = 45.0      # lowered from 50 — smarter scoring is more calibrated
MARKET_OPEN_HOUR    = 9
MARKET_OPEN_MIN     = 35        # start 5 min after open (skip noise)
FORCE_CLOSE_HOUR    = 15
FORCE_CLOSE_MIN     = 50        # liquidate everything
STOP_NEW_HOUR       = 15
STOP_NEW_MIN        = 45        # stop opening new positions
NEWS_POLL_INTERVAL  = 900       # poll news every 15 min
WEAK_SECTORS        = {"Real Estate", "Energy"}

# Adaptive scan intervals (seconds) based on session phase
SCAN_OPENING     = 60    # first 30 min: every 60 seconds
SCAN_POWER_HOUR  = 120   # 3:00-3:45 PM: every 2 minutes
SCAN_DEFAULT     = 300   # normal: every 5 minutes
SCAN_MIDDAY      = 600   # 11:30-2:00 PM lull: every 10 minutes

# ─── Logging ──────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "alpha_intraday.log"),
                           encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("AlphaIntraday")


# ═══════════════════════════════════════════════════════════════════════════
# 1. INTRADAY WALLET
# ═══════════════════════════════════════════════════════════════════════════
class IntradayWallet:
    """
    Tracks intraday virtual portfolio. Similar to VirtualWallet
    but optimized for same-day open/close cycles.
    """

    def __init__(self, path=WALLET_PATH, initial=INITIAL_BALANCE):
        self.path = path
        self.initial = initial
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {
            "initial_balance": self.initial,
            "cash": self.initial,
            "realized_pnl": 0.0,
            "positions": [],
            "trade_history": [],
            "daily_stats": [],
            "last_run_date": None,
            "total_trading_days": 0,
        }

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    @property
    def equity(self):
        return self.state["initial_balance"] + self.state["realized_pnl"]

    @property
    def cash(self):
        return self.state["cash"]

    @property
    def positions(self):
        return self.state["positions"]

    @property
    def n_open(self):
        return len(self.state["positions"])

    def open_position(self, ticker, shares, price, score=0, atr_pct=0.02):
        cost = round(shares * price, 4)
        self.state["cash"] = round(self.state["cash"] - cost, 4)
        self.state["positions"].append({
            "ticker": ticker,
            "shares": round(shares, 6),
            "entry_price": round(price, 4),
            "entry_time": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
            "entry_cost": cost,
            "peak_price": round(price, 4),
            "entry_score": score,
            "atr_pct": atr_pct,
            "partial_exited": False,
        })
        self.save()
        log.info(f"  📗 OPEN  {ticker}: {shares:.4f} sh @ ${price:.2f} "
                 f"= ${cost:.2f} (score: {score})")

    def close_position(self, ticker, exit_price, reason="manual"):
        pos = next((p for p in self.positions if p["ticker"] == ticker), None)
        if not pos:
            log.warning(f"  ⚠️ No position found for {ticker}")
            return 0.0
        exit_value = round(pos["shares"] * exit_price, 4)
        pnl = round(exit_value - pos["entry_cost"], 4)
        self.state["cash"] = round(self.state["cash"] + exit_value, 4)
        self.state["realized_pnl"] = round(self.state["realized_pnl"] + pnl, 4)
        self.state["positions"].remove(pos)
        ret_pct = round(pnl / pos["entry_cost"] * 100, 2) if pos["entry_cost"] else 0
        self.state["trade_history"].append({
            "ticker": ticker,
            "entry_time": pos["entry_time"],
            "exit_time": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": pos["entry_price"],
            "exit_price": round(exit_price, 4),
            "shares": pos["shares"],
            "pnl": pnl,
            "return_pct": ret_pct,
            "exit_reason": reason,
        })
        self.save()
        emoji = "📈" if pnl >= 0 else "📉"
        log.info(f"  {emoji} CLOSE {ticker}: PnL=${pnl:+.4f} ({ret_pct:+.1f}%) "
                 f"| reason: {reason}")
        return pnl

    def partial_close(self, ticker, exit_price, fraction=0.5):
        """Sell a fraction of a position (for scaling exits)."""
        pos = next((p for p in self.positions if p["ticker"] == ticker), None)
        if not pos:
            return 0.0
        sell_shares = round(pos["shares"] * fraction, 6)
        sell_value = round(sell_shares * exit_price, 4)
        sell_cost_portion = round(pos["entry_cost"] * fraction, 4)
        pnl = round(sell_value - sell_cost_portion, 4)

        # Update position
        pos["shares"] = round(pos["shares"] - sell_shares, 6)
        pos["entry_cost"] = round(pos["entry_cost"] - sell_cost_portion, 4)
        pos["partial_exited"] = True

        self.state["cash"] = round(self.state["cash"] + sell_value, 4)
        self.state["realized_pnl"] = round(self.state["realized_pnl"] + pnl, 4)

        ret_pct = round(pnl / sell_cost_portion * 100, 2) if sell_cost_portion else 0
        self.state["trade_history"].append({
            "ticker": ticker,
            "entry_time": pos["entry_time"],
            "exit_time": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": pos["entry_price"],
            "exit_price": round(exit_price, 4),
            "shares": sell_shares,
            "pnl": pnl,
            "return_pct": ret_pct,
            "exit_reason": "partial_exit",
        })
        self.save()
        log.info(f"  📊 PARTIAL {ticker}: sold {sell_shares:.4f} sh @ ${exit_price:.2f} "
                 f"PnL=${pnl:+.4f} ({ret_pct:+.1f}%)")
        return pnl

    def close_all(self, alpaca, reason="force_close"):
        """Force-close all positions. Called at end of day."""
        closed_pnl = 0.0
        for pos in self.positions[:]:
            ticker = pos["ticker"]
            try:
                price = alpaca.get_latest_price(ticker)
                if price is None:
                    continue
                alpaca.sell_position(ticker)
                pnl = self.close_position(ticker, price, reason)
                closed_pnl += pnl
            except Exception as e:
                log.error(f"  ❌ Force-close failed for {ticker}: {e}")
                # Retry once
                try:
                    alpaca.sell_position(ticker)
                except Exception:
                    log.critical(f"  🚨 DOUBLE FAIL: {ticker} may be held overnight!")
        return closed_pnl

    def held_tickers(self):
        return {p["ticker"] for p in self.positions}

    def record_daily_stats(self, daily_pnl, trades_count):
        today = datetime.now(ET).strftime("%Y-%m-%d")
        self.state["daily_stats"].append({
            "date": today,
            "pnl": round(daily_pnl, 4),
            "trades": trades_count,
            "equity": round(self.equity, 4),
        })
        self.state["last_run_date"] = today
        self.state["total_trading_days"] += 1
        self.save()

    def summary(self):
        log.info(f"  💰 Equity:          ${self.equity:.2f}")
        log.info(f"  💵 Cash:            ${self.cash:.2f}")
        log.info(f"  📊 Realized PnL:    ${self.state['realized_pnl']:+.2f}")
        log.info(f"  📂 Open Positions:  {self.n_open}")
        log.info(f"  📜 Total Trades:    {len(self.state['trade_history'])}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ALPACA CLIENT (Extended for Intraday V2)
# ═══════════════════════════════════════════════════════════════════════════
class AlpacaIntradayClient:
    """
    Alpaca REST client extended for intraday trading V2.
    Adds fill verification, smart limit pricing, and position listing.
    """
    PAPER_URL = "https://paper-api.alpaca.markets"
    DATA_URL  = "https://data.alpaca.markets"

    def __init__(self, api_key, secret_key):
        self.base = self.PAPER_URL
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

    def _get(self, url, params=None):
        r = requests.get(url, headers=self.headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, url, data):
        r = requests.post(url, json=data, headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def _delete(self, url):
        r = requests.delete(url, headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def is_market_open(self):
        clock = self._get(f"{self.base}/v2/clock")
        log.info(f"  🕐 Market is_open={clock.get('is_open')} | "
                 f"next_close={clock.get('next_close', '')[:19]}")
        return clock["is_open"]

    def get_account(self):
        return self._get(f"{self.base}/v2/account")

    def get_latest_price(self, ticker):
        try:
            r = self._get(f"{self.DATA_URL}/v2/stocks/{ticker}/trades/latest?feed=iex")
            return float(r["trade"]["p"])
        except Exception as e:
            log.warning(f"  ⚠️ Price failed for {ticker}: {e}")
            return None

    def get_latest_quote(self, ticker):
        """Get latest bid/ask quote for spread checking."""
        try:
            r = self._get(f"{self.DATA_URL}/v2/stocks/{ticker}/quotes/latest?feed=iex")
            quote = r.get("quote", {})
            return {
                "bid": float(quote.get("bp", 0)),
                "ask": float(quote.get("ap", 0)),
                "bid_size": int(quote.get("bs", 0)),
                "ask_size": int(quote.get("as", 0)),
            }
        except Exception:
            return None

    def get_positions(self):
        """Get all current Alpaca positions for reconciliation."""
        try:
            positions = self._get(f"{self.base}/v2/positions")
            return [
                {
                    "ticker": p["symbol"],
                    "shares": float(p["qty"]),
                    "market_value": float(p["market_value"]),
                    "avg_entry_price": float(p["avg_entry_price"]),
                }
                for p in positions
            ]
        except Exception as e:
            log.warning(f"  ⚠️ Failed to get positions: {e}")
            return []

    def get_order_status(self, order_id):
        """Check the fill status of a submitted order."""
        try:
            return self._get(f"{self.base}/v2/orders/{order_id}")
        except Exception:
            return None

    def buy_smart(self, ticker, dollar_amount, quote=None):
        """
        Smart buy: try limit at ask-$0.01 for high fill probability,
        wait briefly, fall back to market if needed.
        Returns (order_response, actual_price, actual_shares) or (None, 0, 0).
        """
        if dollar_amount < 1.0:
            return None, 0, 0

        # Determine limit price: ask - $0.01 (almost guaranteed fill)
        if quote and quote["ask"] > 0:
            limit_price = round(quote["ask"] - 0.01, 2)
            shares = round(dollar_amount / limit_price, 6)

            order = {
                "symbol": ticker,
                "qty": str(shares),
                "side": "buy",           # ☪️ HARDCODED: Long only
                "type": "limit",
                "limit_price": str(limit_price),
                "time_in_force": "ioc",  # Immediate-or-Cancel
            }
            log.info(f"  🛒 SMART BUY {ticker}: {shares:.4f} sh @ limit ${limit_price:.2f}")
            try:
                resp = self._post(f"{self.base}/v2/orders", order)
                order_id = resp.get("id")

                # Check fill after 1 second
                time.sleep(1)
                status = self.get_order_status(order_id)
                if status and status.get("status") in ("filled", "partially_filled"):
                    filled_price = float(status.get("filled_avg_price", limit_price))
                    filled_qty = float(status.get("filled_qty", shares))
                    log.info(f"  ✅ FILLED {ticker}: {filled_qty:.4f} sh @ ${filled_price:.2f}")
                    return status, filled_price, filled_qty

                # If not filled, try market order
                log.info(f"  ⏳ Limit not filled for {ticker}, switching to market...")
            except Exception as e:
                log.warning(f"  ⚠️ Limit order failed for {ticker}: {e}")

        # Fallback: market order with notional
        try:
            order = {
                "symbol": ticker,
                "notional": round(dollar_amount, 2),
                "side": "buy",           # ☪️ HARDCODED: Long only
                "type": "market",
                "time_in_force": "day",
            }
            log.info(f"  🛒 MARKET BUY {ticker}: ${dollar_amount:.2f}")
            resp = self._post(f"{self.base}/v2/orders", order)
            order_id = resp.get("id")

            # Wait for fill
            time.sleep(1.5)
            status = self.get_order_status(order_id)
            if status and status.get("status") == "filled":
                filled_price = float(status.get("filled_avg_price", 0))
                filled_qty = float(status.get("filled_qty", 0))
                log.info(f"  ✅ MARKET FILLED {ticker}: {filled_qty:.4f} sh @ ${filled_price:.2f}")
                return status, filled_price, filled_qty
            elif status:
                # May still be pending — use estimated values
                est_price = quote["ask"] if quote and quote["ask"] > 0 else dollar_amount
                est_shares = dollar_amount / est_price if est_price > 0 else 0
                return status, est_price, est_shares

        except Exception as e:
            log.error(f"  ❌ Market buy failed for {ticker}: {e}")

        return None, 0, 0

    def sell_position(self, ticker):
        log.info(f"  🏷️ SELL {ticker} (close position)")
        return self._delete(f"{self.base}/v2/positions/{ticker}")

    def sell_fraction(self, ticker, shares):
        """Sell specific number of shares (for partial exits)."""
        order = {
            "symbol": ticker,
            "qty": str(round(shares, 6)),
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
        }
        log.info(f"  🏷️ PARTIAL SELL {ticker}: {shares:.4f} sh")
        return self._post(f"{self.base}/v2/orders", order)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ADAPTIVE SCAN INTERVAL
# ═══════════════════════════════════════════════════════════════════════════
def get_scan_interval():
    """
    Return the appropriate scan interval based on market session phase.
    Opening and closing periods = faster scanning for opportunities.
    """
    now = datetime.now(ET)
    market_open = now.replace(hour=9, minute=30, second=0)
    mins_since_open = max(0, (now - market_open).total_seconds() / 60)

    if mins_since_open < 30:
        return SCAN_OPENING       # 60s — opening sprint
    elif mins_since_open >= 330:   # 3:00 PM+
        return SCAN_POWER_HOUR    # 120s — power hour
    elif 120 < mins_since_open < 270:
        return SCAN_MIDDAY        # 600s — midday lull
    else:
        return SCAN_DEFAULT       # 300s — normal


# ═══════════════════════════════════════════════════════════════════════════
# 4. SINGLE SCAN CYCLE (V2 — batch scanning + smart execution)
# ═══════════════════════════════════════════════════════════════════════════
def run_scan_cycle(alpaca, wallet, risk_mgr, feature_engine, news_fetcher,
                   fast_scanner, halal_tickers, fundamentals,
                   intraday_model=None, model_features=None):
    """
    Execute one scan cycle (V2):
      1. Check time constraints
      2. Monitor & exit open positions (ATR stops, scaling exits, time stops)
      3. Batch-scan universe for new entries
      4. Execute entries with smart order handling
    """
    now = datetime.now(ET)
    log.info(f"\n{'─'*55}")
    log.info(f"⏱️ SCAN CYCLE @ {now.strftime('%H:%M:%S ET')}")

    # ── Time checks ───────────────────────────────────────────────────
    force_close_time = now.replace(hour=FORCE_CLOSE_HOUR, minute=FORCE_CLOSE_MIN,
                                   second=0)
    stop_new_time = now.replace(hour=STOP_NEW_HOUR, minute=STOP_NEW_MIN,
                                second=0)

    # Force-close all at 3:50 PM
    if now >= force_close_time:
        log.info("🔔 FORCE-CLOSE TIME — liquidating all positions")
        pnl = wallet.close_all(alpaca, reason="eod_close")
        risk_mgr.record_trade_result(pnl, wallet.equity)
        risk_mgr.intraday_summary()
        wallet.summary()
        return "CLOSED"

    # ── Phase 1: Monitor open positions (ATR stops + scaling exits) ───
    if wallet.n_open > 0:
        log.info(f"📊 Monitoring {wallet.n_open} open positions...")

        # Build ATR values dict from position data
        atr_values = {}
        for pos in wallet.positions:
            atr_pct = pos.get("atr_pct", 0.02)
            atr_values[pos["ticker"]] = atr_pct * pos["entry_price"]

        exits = risk_mgr.check_intraday_exits(
            wallet.positions,
            lambda t: alpaca.get_latest_price(t),
            atr_values=atr_values
        )

        for ticker, reason in exits:
            try:
                price = alpaca.get_latest_price(ticker)
                if price is None:
                    continue

                if reason == "partial_exit":
                    # Sell 50% of position
                    pos = next((p for p in wallet.positions if p["ticker"] == ticker), None)
                    if pos:
                        sell_shares = round(pos["shares"] * 0.5, 6)
                        alpaca.sell_fraction(ticker, sell_shares)
                        pnl = wallet.partial_close(ticker, price, 0.5)
                        risk_mgr.record_trade_result(pnl, wallet.equity)
                else:
                    # Full close
                    alpaca.sell_position(ticker)
                    pnl = wallet.close_position(ticker, price, reason)
                    risk_mgr.record_trade_result(pnl, wallet.equity)
            except Exception as e:
                log.error(f"  ❌ Exit failed for {ticker}: {e}")

    # ── Phase 2: Check if we can trade ────────────────────────────────
    if now >= stop_new_time:
        log.info("⏰ Past 3:45 PM — no new entries, monitoring only")
        return "MONITORING"

    can_trade, reason = risk_mgr.can_open_new_trade()
    if not can_trade:
        log.info(f"🚫 Cannot open new trades: {reason}")
        return "BLOCKED"

    # Use smart position sizing to get max positions for this account size
    sizing = risk_mgr.calculate_position_size_v2(wallet.equity, 100, 0.5)
    max_positions = sizing["max_positions"]

    slots = max_positions - wallet.n_open
    if slots <= 0:
        log.info(f"📦 All {max_positions} slots filled. Monitoring only.")
        return "FULL"

    cash = wallet.cash
    if cash < 5.0:
        log.info(f"💸 Insufficient cash (${cash:.2f}). Monitoring only.")
        return "NO_CASH"

    # ── Phase 3: BATCH scan universe ──────────────────────────────────
    log.info(f"🔍 Batch scanning {len(halal_tickers)} Halal stocks...")
    scan_start = time.time()

    # Poll news (rate-limited by news_fetcher internally)
    try:
        news_fetcher.poll(halal_tickers[:50])
    except Exception as e:
        log.warning(f"  News poll error: {e}")

    held = wallet.held_tickers()

    # Get scan order (watchlist first)
    ordered_tickers = fast_scanner.get_scan_order(halal_tickers)

    # Filter out held tickers and weak sectors
    scan_tickers = []
    for ticker in ordered_tickers:
        if ticker in held:
            continue
        fund = fundamentals.get(ticker, {})
        if fund.get("Sector", "") in WEAK_SECTORS:
            continue
        scan_tickers.append(ticker)

    # Batch fetch latest quotes for all candidates
    batch_quotes = {}
    try:
        batch_quotes = fast_scanner.fetch_batch_quotes(scan_tickers)
    except Exception as e:
        log.warning(f"  ⚠️ Batch quotes failed: {e}")

    # Score candidates — use individual feature computation
    # (batch bars already cached from previous cycles via fast_scanner)
    candidates = []
    all_scores = []
    scanned = 0
    skipped_features = 0

    for ticker in scan_tickers:
        # Rate limit: slight pause to avoid 429 (batch quotes already done)
        if scanned > 0 and scanned % 40 == 0:
            time.sleep(0.3)  # brief pause every 40 tickers

        # Build features
        features = feature_engine.build_features(ticker)
        if features is None:
            skipped_features += 1
            continue

        scanned += 1

        # Get catalyst score from news
        catalyst, news_details = news_fetcher.get_catalyst_score(ticker)

        # Compute entry score (returns (score, details_dict) tuple)
        score, score_details = feature_engine.compute_entry_score(features, catalyst)

        # ── ML model boost: if intraday model is loaded, blend its prediction ──
        if intraday_model is not None and model_features is not None:
            try:
                fv = np.array([[features.get(f, 0) for f in model_features]],
                              dtype=np.float32)
                ml_prob = float(intraday_model.predict(fv)[0])
                # Blend: 60% rule-based + 40% ML (scaled to 0-100)
                score = round(score * 0.6 + ml_prob * 100 * 0.4, 1)
            except Exception:
                pass  # fall back to rule-based score

        # Track all scores for debugging
        all_scores.append({"ticker": ticker, "score": score, "price": features["price"]})

        if score >= MIN_ENTRY_SCORE:
            candidates.append({
                "ticker": ticker,
                "score": score,
                "score_details": score_details,
                "price": features["price"],
                "features": features,
                "atr_pct": score_details.get("atr_pct", 0.02),
                "catalyst": catalyst,
                "news": news_details,
            })

        # Stop scanning after enough candidates found
        if len(candidates) >= slots * 4:
            break

    scan_time = time.time() - scan_start

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Update watchlist with all scored tickers
    fast_scanner.update_watchlist(
        {s["ticker"]: s["score"] for s in all_scores if s["score"] > 20}
    )

    # Log scan results
    all_scores.sort(key=lambda x: x["score"], reverse=True)
    log.info(f"  📊 Scan: {scanned} evaluated, {skipped_features} skipped "
             f"in {scan_time:.1f}s")
    log.info(f"  📊 TOP 10 SCORES (threshold = {MIN_ENTRY_SCORE}):")
    for s in all_scores[:10]:
        passed = "✅" if s["score"] >= MIN_ENTRY_SCORE else "❌"
        log.info(f"     {passed} {s['ticker']:6s} → score={s['score']:5.1f}  "
                 f"(${s['price']:.2f})")

    if not candidates:
        log.info(f"  ℹ️ No stocks passed entry threshold ({MIN_ENTRY_SCORE}).")
        if all_scores:
            best = all_scores[0]
            log.info(f"  💡 Closest: {best['ticker']} score={best['score']:.1f} "
                     f"(needs +{MIN_ENTRY_SCORE - best['score']:.1f})")
        return "NO_SIGNALS"

    log.info(f"  ✅ {len(candidates)} candidates passed (min score: {MIN_ENTRY_SCORE})")
    for c in candidates[:5]:
        details = c.get("score_details", {})
        log.info(f"     {c['ticker']:6s} → score={c['score']:.1f} "
                 f"(VWAP={details.get('vwap_score', 0):.0f} "
                 f"Mom={details.get('momentum_score', 0):.0f} "
                 f"Vol={details.get('volume_score', 0):.0f} "
                 f"Cat={c['catalyst']:.2f})")

    # ── Phase 4: Execute entries with smart ordering ──────────────────
    bought = 0
    for c in candidates[:slots]:
        ticker = c["ticker"]
        price = c["price"]
        score = c["score"]
        atr_pct = c["atr_pct"]

        # Use batch quote if available, otherwise fetch individually
        quote = batch_quotes.get(ticker)
        if not quote or quote.get("bid", 0) <= 0:
            quote = alpaca.get_latest_quote(ticker)

        # Spread check
        if quote and quote.get("bid", 0) > 0 and quote.get("ask", 0) > 0:
            if not risk_mgr.passes_spread_check(quote["bid"], quote["ask"], price):
                log.info(f"  🚫 {ticker}: spread too wide, skipping")
                continue

        # Calculate position size with ATR-aware sizing
        atr_dollar = atr_pct * price
        sizing = risk_mgr.calculate_position_size_v2(
            wallet.equity, price, atr_dollar
        )
        dollar_amount = min(sizing["dollar_amount"], cash)

        if dollar_amount < 1.0:
            log.info(f"  💸 Not enough for {ticker} (${dollar_amount:.2f})")
            continue

        try:
            # Smart buy with fill verification
            order_resp, fill_price, fill_shares = alpaca.buy_smart(
                ticker, dollar_amount, quote
            )
            if order_resp is None or fill_shares <= 0:
                log.warning(f"  ⚠️ No fill for {ticker}")
                continue

            # Use ACTUAL fill price, not stale quote
            wallet.open_position(ticker, fill_shares, fill_price, score, atr_pct)
            cash -= fill_shares * fill_price
            bought += 1

        except Exception as e:
            log.error(f"  ❌ Buy failed for {ticker}: {e}")

    log.info(f"  📊 Cycle complete: {bought} new positions opened")
    wallet.summary()
    return "OK"


# ═══════════════════════════════════════════════════════════════════════════
# 5. POSITION RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════════
def reconcile_on_startup(alpaca, wallet, risk_mgr):
    """
    On startup, verify wallet positions match Alpaca's actual positions.
    Fix any discrepancies caused by crashes or missed order confirmations.
    """
    log.info("🔄 Reconciling wallet with Alpaca positions...")

    alpaca_positions = alpaca.get_positions()
    result = risk_mgr.reconcile_positions(
        wallet.positions, alpaca_positions
    )

    # Fix orphaned wallet positions (we think we own, but Alpaca says no)
    for ticker in result.get("orphaned_wallet", []):
        log.warning(f"  🗑️ Removing phantom position {ticker} from wallet")
        pos = next((p for p in wallet.positions if p["ticker"] == ticker), None)
        if pos:
            wallet.state["positions"].remove(pos)

    # Fix orphaned Alpaca positions (Alpaca has, we don't know about)
    for ticker in result.get("orphaned_alpaca", []):
        log.warning(f"  📥 Adding ghost position {ticker} to wallet from Alpaca")
        alp_pos = next((p for p in alpaca_positions if p["ticker"] == ticker), None)
        if alp_pos:
            wallet.open_position(
                ticker, alp_pos["shares"],
                alp_pos["avg_entry_price"], score=0
            )

    wallet.save()


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AlphaShariaBot Intraday Engine V2")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuous scan loop (default: single scan)")
    parser.add_argument("--force-close", action="store_true",
                        help="Force-close all open positions and exit")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("⚡ AlphaShariaBot — Intraday Day Trading Engine V2")
    log.info("=" * 60)

    # ── Load API keys ─────────────────────────────────────────────────
    load_dotenv(ENV_PATH)
    api_key = os.getenv("ALPACA_API_KEY")
    secret  = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        log.error("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        return

    # ── Initialize all components ─────────────────────────────────────
    alpaca = AlpacaIntradayClient(api_key, secret)
    wallet = IntradayWallet()
    risk_mgr = IntradayRiskManager()
    feature_engine = IntradayFeatureEngine(alpaca)
    news_fetcher = IntradayNewsFetcher(api_key, secret)
    fast_scanner = FastScanner(api_key, secret)
    premarket = PremarketScanner(api_key, secret)
    monitor = MarketMonitor(api_key, secret)

    # ── Load intraday ML model (optional) ─────────────────────────────
    intraday_model = None
    model_features = None
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        try:
            import lightgbm as lgb
            intraday_model = lgb.Booster(model_file=MODEL_PATH)
            with open(FEATURES_PATH) as f:
                model_features = json.load(f)
            # Load metrics for logging
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH) as f:
                    metrics = json.load(f)
                log.info(f"  🤖 ML model loaded: AUC={metrics.get('auc', '?')} "
                         f"Top10%={metrics.get('top_10pct_precision', '?')} "
                         f"({len(model_features)} features)")
            else:
                log.info(f"  🤖 ML model loaded ({len(model_features)} features)")
        except ImportError:
            log.warning("  ⚠️ LightGBM not installed — running rule-based only")
        except Exception as e:
            log.warning(f"  ⚠️ Failed to load ML model: {e} — running rule-based only")
    else:
        log.info("  ℹ️ No intraday ML model found — using rule-based scoring")
        log.info(f"     Train one with: python scripts/train_intraday_model.py")

    log.info(f"  💰 Account equity: ${wallet.equity:.2f}")

    # Force-close mode
    if args.force_close:
        log.info("🔔 Force-closing all positions...")
        wallet.close_all(alpaca, reason="manual_force_close")
        wallet.summary()
        return

    # Load halal universe
    halal_tickers = []
    if os.path.exists(HALAL_CSV):
        halal_tickers = pd.read_csv(HALAL_CSV)["ticker"].str.upper().tolist()
        halal_tickers = [t for t in halal_tickers
                         if t.isalpha() and t != "XYZ"]
    log.info(f"  ☪️ Halal universe: {len(halal_tickers)} tickers")

    # Load fundamentals
    fundamentals = {}
    if os.path.exists(FUND_CSV):
        fund_df = pd.read_csv(FUND_CSV)
        col = "Ticker" if "Ticker" in fund_df.columns else "ticker"
        for _, row in fund_df.iterrows():
            fundamentals[row[col]] = row.to_dict()

    # ── Market check ──────────────────────────────────────────────────
    try:
        now_et = datetime.now(ET)
        log.info(f"  🕐 ET time: {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
        market_open = alpaca.is_market_open()
        if not market_open:
            log.info("🔒 Market is closed. Nothing to do.")
            wallet.summary()
            return
        else:
            log.info("✅ Market is OPEN!")
    except Exception as e:
        log.error(f"❌ Cannot reach Alpaca API: {e}")
        import traceback
        log.error(traceback.format_exc())
        return

    # ── Position reconciliation ───────────────────────────────────────
    reconcile_on_startup(alpaca, wallet, risk_mgr)

    # Reset daily counters
    risk_mgr.reset_daily(wallet.equity)

    log.info("\n📋 WALLET STATUS (Start of Day)")
    wallet.summary()

    # ── Pre-market gap scan ───────────────────────────────────────────
    now = datetime.now(ET)
    mins_since_open = (now - now.replace(hour=9, minute=30, second=0)).total_seconds() / 60
    if mins_since_open < 15:
        log.info("🌅 Running pre-market gap scan...")
        try:
            gap_result = premarket.get_gap_watchlist(halal_tickers, min_gap_pct=0.005)
            gap_ups = gap_result.get("gap_ups", [])
            gap_downs = gap_result.get("gap_downs", [])
            if gap_ups:
                log.info(f"  📈 Gap-ups ({len(gap_ups)}):")
                for g in gap_ups[:8]:
                    log.info(f"     {g['ticker']:6s} +{g['gap_pct']*100:.1f}% "
                             f"vol_ratio={g.get('volume_ratio', 0):.1f}")
                # Add gap-ups to watchlist for priority scanning
                fast_scanner.update_watchlist(
                    {g["ticker"]: 50 + g["gap_pct"] * 1000 for g in gap_ups}
                )
            if gap_downs:
                log.info(f"  📉 Gap-downs ({len(gap_downs)}) — avoiding:")
                for g in gap_downs[:5]:
                    log.info(f"     {g['ticker']:6s} {g['gap_pct']*100:+.1f}%")
        except Exception as e:
            log.warning(f"  ⚠️ Pre-market scan failed: {e}")

    if args.loop:
        # ── Continuous loop mode ──────────────────────────────────────
        log.info(f"\n🔄 Starting adaptive scan loop...")

        # Start real-time market monitor for watchlist
        if halal_tickers:
            top_watch = fast_scanner.get_scan_order(halal_tickers)[:50]
            monitor.set_watchlist(top_watch)
            monitor.start()
            log.info(f"  📡 Real-time monitor started ({len(top_watch)} tickers)")
        last_news_poll = 0

        while True:
            now = datetime.now(ET)
            # Check if market is still open
            market_close = now.replace(hour=16, minute=0, second=0)
            if now >= market_close:
                log.info("🔔 Market closed. Final force-close...")
                wallet.close_all(alpaca, reason="eod_close")
                wallet.record_daily_stats(risk_mgr.daily_pnl,
                                          risk_mgr.daily_trades)
                risk_mgr.intraday_summary()
                wallet.summary()
                break

            status = run_scan_cycle(alpaca, wallet, risk_mgr,
                                     feature_engine, news_fetcher,
                                     fast_scanner, halal_tickers,
                                     fundamentals,
                                     intraday_model, model_features)

            if status == "CLOSED":
                wallet.record_daily_stats(risk_mgr.daily_pnl,
                                          risk_mgr.daily_trades)
                break

            # ── News-driven fast entry: check for urgent news ─────────
            urgent_tickers = news_fetcher.get_urgent_tickers(threshold=0.5)
            if urgent_tickers:
                log.info(f"  📰 URGENT NEWS on {urgent_tickers} — fast scan!")
                status = run_scan_cycle(alpaca, wallet, risk_mgr,
                                         feature_engine, news_fetcher,
                                         fast_scanner, urgent_tickers,
                                         fundamentals,
                                         intraday_model, model_features)

            # ── Market monitor events: check for real-time triggers ───
            event_tickers = monitor.get_event_tickers(clear=True)
            if event_tickers:
                log.info(f"  📡 MARKET EVENTS on {event_tickers} — fast scan!")
                status = run_scan_cycle(alpaca, wallet, risk_mgr,
                                         feature_engine, news_fetcher,
                                         fast_scanner, event_tickers,
                                         fundamentals,
                                         intraday_model, model_features)

            interval = get_scan_interval()
            log.info(f"  ⏳ Next scan in {interval}s "
                     f"(adaptive: {_interval_label(interval)})")
            time.sleep(interval)
    else:
        # ── Single scan mode ──────────────────────────────────────────
        run_scan_cycle(alpaca, wallet, risk_mgr, feature_engine,
                       news_fetcher, fast_scanner, halal_tickers,
                       fundamentals, intraday_model, model_features)

    log.info("=" * 60)
    log.info("✅ Intraday session complete.")


def _interval_label(interval):
    """Human-readable label for scan interval."""
    labels = {
        SCAN_OPENING: "opening sprint",
        SCAN_POWER_HOUR: "power hour",
        SCAN_MIDDAY: "midday lull",
        SCAN_DEFAULT: "normal",
    }
    return labels.get(interval, f"{interval}s")


if __name__ == "__main__":
    main()
