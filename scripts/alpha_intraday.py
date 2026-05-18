"""
alpha_intraday.py — Intraday Day Trading Engine for AlphaShariaBot
===================================================================
Replaces alpha_live.py as the primary engine for day trading mode.
Scans for opportunities every 5 minutes during market hours,
manages positions with tight stop-losses and take-profits,
and force-closes all positions before market close.

Key differences from alpha_live.py (swing trading):
  - Multiple trades per day (vs. once daily)
  - Positions held minutes to hours (vs. 10 days)
  - All positions closed before 3:50 PM ET (no overnight risk)
  - Rule-based entries using VWAP + momentum + volume + news catalyst
  - Tighter risk controls: -0.8% SL, +1.5% TP, -3% daily loss limit
  - Spread-aware execution with limit orders

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

import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_manager import IntradayRiskManager
from intraday_features import IntradayFeatureEngine
from realtime_news import IntradayNewsFetcher

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALLET_PATH    = os.path.join(BASE_DIR, "data", "live", "intraday_wallet.json")
HALAL_CSV      = os.path.join(BASE_DIR, "data", "halal_stocks.csv")
FUND_CSV       = os.path.join(BASE_DIR, "data", "fundamentals.csv")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
ENV_PATH       = os.path.join(BASE_DIR, ".env")
ET             = ZoneInfo("America/New_York")

# ─── Strategy Constants ───────────────────────────────────────────────────
INITIAL_BALANCE     = 1000.0
MAX_INTRADAY_POS    = 12       # max concurrent positions
MIN_ENTRY_SCORE     = 55.0     # minimum rule-based score to enter
SCAN_INTERVAL_SEC   = 300      # 5 minutes between scans
MARKET_OPEN_HOUR    = 9
MARKET_OPEN_MIN     = 35       # start 5 min after open (skip noise)
FORCE_CLOSE_HOUR    = 15
FORCE_CLOSE_MIN     = 50       # liquidate everything
STOP_NEW_HOUR       = 15
STOP_NEW_MIN        = 45       # stop opening new positions
NEWS_POLL_INTERVAL  = 900      # poll news every 15 min
WEAK_SECTORS        = {"Real Estate", "Energy"}

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

    def open_position(self, ticker, shares, price, score=0):
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
        log.info(f"  📂 Open Positions:  {self.n_open}/{MAX_INTRADAY_POS}")
        log.info(f"  📜 Total Trades:    {len(self.state['trade_history'])}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ALPACA CLIENT (Extended for Intraday)
# ═══════════════════════════════════════════════════════════════════════════
class AlpacaIntradayClient:
    """
    Alpaca REST client extended for intraday trading.
    Adds quote fetching and limit order support.
    """
    PAPER_URL = "https://paper-api.alpaca.markets"
    DATA_URL  = "https://data.alpaca.markets"

    def __init__(self, api_key, secret_key):
        self.base = self.PAPER_URL
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

    def _get(self, url):
        r = requests.get(url, headers=self.headers, timeout=15)
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
        log.info(f"  🕐 Alpaca Clock API response:")
        log.info(f"     is_open:    {clock.get('is_open')}")
        log.info(f"     timestamp:  {clock.get('timestamp')}")
        log.info(f"     next_open:  {clock.get('next_open')}")
        log.info(f"     next_close: {clock.get('next_close')}")
        return clock["is_open"]

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

    def buy_limit(self, ticker, shares, limit_price):
        """Submit a limit buy order with IOC time-in-force."""
        if shares < 0.001:
            return None
        order = {
            "symbol": ticker,
            "qty": str(round(shares, 6)),
            "side": "buy",           # ☪️ HARDCODED: Long only
            "type": "limit",
            "limit_price": str(round(limit_price, 2)),
            "time_in_force": "ioc",  # Immediate-or-Cancel
        }
        log.info(f"  🛒 LIMIT BUY {ticker}: {shares:.4f} sh @ ${limit_price:.2f}")
        return self._post(f"{self.base}/v2/orders", order)

    def buy_fractional(self, ticker, dollar_amount):
        """Fallback: market order with notional amount."""
        if dollar_amount < 1.0:
            return None
        order = {
            "symbol": ticker,
            "notional": round(dollar_amount, 2),
            "side": "buy",           # ☪️ HARDCODED: Long only
            "type": "market",
            "time_in_force": "day",
        }
        log.info(f"  🛒 MARKET BUY {ticker}: ${dollar_amount:.2f}")
        return self._post(f"{self.base}/v2/orders", order)

    def sell_position(self, ticker):
        log.info(f"  🏷️ SELL {ticker} (close position)")
        return self._delete(f"{self.base}/v2/positions/{ticker}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. SINGLE SCAN CYCLE
# ═══════════════════════════════════════════════════════════════════════════
def run_scan_cycle(alpaca, wallet, risk_mgr, feature_engine, news_fetcher,
                   halal_tickers, fundamentals):
    """
    Execute one scan cycle:
      1. Check time constraints
      2. Monitor & exit open positions
      3. Scan for new entries
      4. Execute entries
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

    # ── Phase 1: Monitor open positions ───────────────────────────────
    if wallet.n_open > 0:
        log.info(f"📊 Monitoring {wallet.n_open} open positions...")
        exits = risk_mgr.check_intraday_exits(
            wallet.positions,
            lambda t: alpaca.get_latest_price(t)
        )
        for ticker, reason in exits:
            try:
                price = alpaca.get_latest_price(ticker)
                if price is None:
                    continue
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

    slots = MAX_INTRADAY_POS - wallet.n_open
    if slots <= 0:
        log.info(f"📦 All {MAX_INTRADAY_POS} slots filled. Monitoring only.")
        return "FULL"

    cash = wallet.cash
    if cash < 5.0:
        log.info(f"💸 Insufficient cash (${cash:.2f}). Monitoring only.")
        return "NO_CASH"

    # ── Phase 3: Poll news & scan candidates ──────────────────────────
    log.info(f"🔍 Scanning {len(halal_tickers)} Halal stocks for entries...")

    # Poll news (rate-limited by news_fetcher internally)
    try:
        news_fetcher.poll(halal_tickers[:50])
    except Exception as e:
        log.warning(f"  News poll error: {e}")

    held = wallet.held_tickers()
    candidates = []

    for ticker in halal_tickers:
        if ticker in held:
            continue

        # Skip weak sectors
        fund = fundamentals.get(ticker, {})
        if fund.get("Sector", "") in WEAK_SECTORS:
            continue

        # Build features
        features = feature_engine.build_features(ticker)
        if features is None:
            continue

        # Get catalyst score from news
        catalyst, news_details = news_fetcher.get_catalyst_score(ticker)

        # Compute entry score
        score = feature_engine.compute_entry_score(features, catalyst)

        if score >= MIN_ENTRY_SCORE:
            candidates.append({
                "ticker": ticker,
                "score": score,
                "price": features["price"],
                "features": features,
                "catalyst": catalyst,
                "news": news_details,
            })

        # Don't scan the entire universe every cycle — stop after enough
        if len(candidates) >= slots * 3:
            break

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    if not candidates:
        log.info("  ℹ️ No stocks passed entry threshold this cycle.")
        return "NO_SIGNALS"

    log.info(f"  ✅ {len(candidates)} candidates passed (min score: {MIN_ENTRY_SCORE})")
    for c in candidates[:5]:
        log.info(f"     {c['ticker']:6s} → score={c['score']:.1f} "
                 f"(catalyst={c['catalyst']:.2f})")

    # ── Phase 4: Execute entries ──────────────────────────────────────
    bought = 0
    for c in candidates[:slots]:
        ticker = c["ticker"]
        price = c["price"]
        score = c["score"]

        # Get quote for spread check
        quote = alpaca.get_latest_quote(ticker)
        if quote and quote["bid"] > 0 and quote["ask"] > 0:
            if not risk_mgr.passes_spread_check(quote["bid"], quote["ask"], price):
                log.info(f"  🚫 {ticker}: spread too wide, skipping")
                continue
            mid_price = (quote["bid"] + quote["ask"]) / 2
        else:
            mid_price = price  # fallback

        # Calculate position size (risk-based)
        stop_price = price * (1 + risk_mgr.config["stop_loss_pct"])
        dollar_amount = risk_mgr.calculate_position_size(
            wallet.equity, price, stop_price
        )
        dollar_amount = min(dollar_amount, cash)

        if dollar_amount < 1.0:
            log.info(f"  💸 Not enough for {ticker} (${dollar_amount:.2f})")
            continue

        try:
            # Try limit order at mid-price first
            shares = round(dollar_amount / mid_price, 6)
            order = alpaca.buy_limit(ticker, shares, mid_price)
            if order is None:
                # Fallback to market order
                order = alpaca.buy_fractional(ticker, dollar_amount)
                if order is None:
                    continue

            wallet.open_position(ticker, shares, price, score)
            cash -= dollar_amount
            bought += 1
            log.info(f"  📗 BOUGHT {ticker}: ${dollar_amount:.2f} "
                     f"(score={score:.1f})")
        except Exception as e:
            log.error(f"  ❌ Buy failed for {ticker}: {e}")

    log.info(f"  📊 Cycle complete: {bought} new positions opened")
    wallet.summary()
    return "OK"


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AlphaShariaBot Intraday Engine")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuous scan loop (default: single scan)")
    parser.add_argument("--force-close", action="store_true",
                        help="Force-close all open positions and exit")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("⚡ AlphaShariaBot — Intraday Day Trading Engine")
    log.info("=" * 60)

    # ── Load API keys ─────────────────────────────────────────────────
    load_dotenv(ENV_PATH)
    api_key = os.getenv("ALPACA_API_KEY")
    secret  = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        log.error("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        return

    alpaca = AlpacaIntradayClient(api_key, secret)
    wallet = IntradayWallet()
    risk_mgr = IntradayRiskManager()
    feature_engine = IntradayFeatureEngine(alpaca)
    news_fetcher = IntradayNewsFetcher(api_key, secret)

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
        log.info(f"  🕐 Local ET time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        log.info(f"  🕐 Weekday: {now_et.strftime('%A')} (0=Mon..6=Sun: {now_et.weekday()})")
        market_open = alpaca.is_market_open()
        if not market_open:
            log.info("🔒 Market is closed according to Alpaca. Nothing to do.")
            log.info("   Possible reasons: weekend, US holiday, or outside 9:30AM-4:00PM ET")
            wallet.summary()
            return
        else:
            log.info("✅ Market is OPEN! Proceeding with scan...")
    except Exception as e:
        log.error(f"❌ Cannot reach Alpaca API: {e}")
        import traceback
        log.error(traceback.format_exc())
        return

    # Reset daily counters
    risk_mgr.reset_daily(wallet.equity)

    log.info("\n📋 WALLET STATUS (Start of Day)")
    wallet.summary()

    if args.loop:
        # ── Continuous loop mode ──────────────────────────────────────
        log.info(f"\n🔄 Starting continuous scan loop "
                 f"(every {SCAN_INTERVAL_SEC}s)...")
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
                                     halal_tickers, fundamentals)

            if status == "CLOSED":
                wallet.record_daily_stats(risk_mgr.daily_pnl,
                                          risk_mgr.daily_trades)
                break

            log.info(f"  ⏳ Next scan in {SCAN_INTERVAL_SEC}s...")
            time.sleep(SCAN_INTERVAL_SEC)
    else:
        # ── Single scan mode ──────────────────────────────────────────
        run_scan_cycle(alpaca, wallet, risk_mgr, feature_engine,
                       news_fetcher, halal_tickers, fundamentals)

    log.info("=" * 60)
    log.info("✅ Intraday session complete.")


if __name__ == "__main__":
    main()
