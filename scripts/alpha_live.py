"""
alpha_live.py — Production Paper Trading Engine for AlphaShariaBot
===================================================================
Virtual $100 wallet on Alpaca's $100K paper account.
Fractional shares, Sharia-compliant (long-only, US equity).
10-day holding period matching the V8 LambdaRank training horizon.

Usage:
    # First run / manual:
    python scripts/alpha_live.py

    # Cron (weekdays 10:00 AM ET):
    0 10 * * 1-5 cd /home/yousef/projects/AlphaShariaBot && ./venv/bin/python scripts/alpha_live.py >> logs/alpha_live.log 2>&1
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import gc
import json
import logging
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_manager import RiskManager

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALLET_PATH    = os.path.join(BASE_DIR, "data", "live", "virtual_wallet.json")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "ranker_model.txt")
FEATURES_PATH  = os.path.join(BASE_DIR, "models", "ranker_features.json")
HALAL_CSV      = os.path.join(BASE_DIR, "data", "halal_stocks.csv")
FUND_CSV       = os.path.join(BASE_DIR, "data", "fundamentals.csv")
SENTIMENT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "daily_sentiment.parquet")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
ENV_PATH       = os.path.join(BASE_DIR, ".env")

# ─── Strategy Constants ───────────────────────────────────────────────────
INITIAL_BALANCE  = 1000.0
MAX_POSITIONS    = 50
PER_POSITION     = 20.0     # fixed $20 per deal
HOLDING_DAYS     = 10       # must match FORWARD_DAYS in training
TOP_K            = 50       # pick top 50 from ranked universe
LOOKBACK_DAYS    = 400      # calendar days → ~280 trading days (need 252 for rolling features)
WEAK_SECTORS     = {"Real Estate", "Energy"}  # excluded from training → exclude from picks

# ─── Logging ──────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "alpha_live.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("AlphaLive")


# ═══════════════════════════════════════════════════════════════════════════
# 1. VIRTUAL WALLET
# ═══════════════════════════════════════════════════════════════════════════
class VirtualWallet:
    """
    Tracks a virtual $100 portfolio independently of Alpaca's $100K balance.
    All position sizing uses THIS balance, never Alpaca's buying power.
    State persisted to JSON after every mutation.
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
            "last_run_date": None,
        }

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    # ── Properties ────────────────────────────────────────────────────
    @property
    def virtual_balance(self):
        """Initial + all realized PnL. This is 'our' total equity."""
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

    # ── Mutations ─────────────────────────────────────────────────────
    def open_position(self, ticker, shares, price):
        cost = round(shares * price, 4)
        self.state["cash"] = round(self.state["cash"] - cost, 4)
        self.state["positions"].append({
            "ticker": ticker,
            "shares": round(shares, 6),
            "entry_price": round(price, 4),
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_cost": cost,
            "trading_days_held": 0,
        })
        self.save()
        log.info(f"  📗 OPEN  {ticker}: {shares:.4f} shares @ ${price:.2f} = ${cost:.2f}")

    def close_position(self, ticker, exit_price):
        pos = next((p for p in self.positions if p["ticker"] == ticker), None)
        if not pos:
            log.warning(f"  ⚠️ No position found for {ticker}")
            return 0.0
        exit_value = round(pos["shares"] * exit_price, 4)
        pnl = round(exit_value - pos["entry_cost"], 4)
        self.state["cash"] = round(self.state["cash"] + exit_value, 4)
        self.state["realized_pnl"] = round(self.state["realized_pnl"] + pnl, 4)
        self.state["positions"].remove(pos)
        self.state["trade_history"].append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "exit_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_price": pos["entry_price"],
            "exit_price": round(exit_price, 4),
            "shares": pos["shares"],
            "pnl": pnl,
            "return_pct": round(pnl / pos["entry_cost"] * 100, 2) if pos["entry_cost"] else 0,
        })
        self.save()
        emoji = "📈" if pnl >= 0 else "📉"
        log.info(f"  {emoji} CLOSE {ticker}: PnL=${pnl:+.4f} "
                 f"({pnl/pos['entry_cost']*100:+.1f}%) | held {pos['trading_days_held']}d")
        return pnl

    def increment_holding_days(self):
        for pos in self.positions:
            pos["trading_days_held"] = pos.get("trading_days_held", 0) + 1
        self.save()

    def expired_positions(self):
        return [p for p in self.positions
                if p.get("trading_days_held", 0) >= HOLDING_DAYS]

    def available_cash_for_new(self):
        """Cash available for new positions."""
        return max(0, self.state["cash"])

    def already_ran_today(self):
        return self.state.get("last_run_date") == datetime.now().strftime("%Y-%m-%d")

    def mark_ran_today(self):
        self.state["last_run_date"] = datetime.now().strftime("%Y-%m-%d")
        self.save()

    def held_tickers(self):
        return {p["ticker"] for p in self.positions}

    def summary(self):
        log.info(f"  💰 Virtual Balance: ${self.virtual_balance:.2f}")
        log.info(f"  💵 Cash:            ${self.cash:.2f}")
        log.info(f"  📊 Realized PnL:    ${self.state['realized_pnl']:+.2f}")
        log.info(f"  📂 Open Positions:  {self.n_open}/{MAX_POSITIONS}")
        log.info(f"  📜 Total Trades:    {len(self.state['trade_history'])}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ALPACA CLIENT (Sharia-Hardened)
# ═══════════════════════════════════════════════════════════════════════════
class AlpacaClient:
    """
    Thin REST wrapper. HARDCODED Sharia constraints:
      - side='buy' only (no shorting)
      - asset_class='us_equity' only (no options/crypto)
      - Fractional shares via 'notional' orders
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

    # ── Market Status ─────────────────────────────────────────────────
    def is_market_open(self):
        clock = self._get(f"{self.base}/v2/clock")
        return clock["is_open"]

    # ── Orders ────────────────────────────────────────────────────────
    def buy_fractional(self, ticker, dollar_amount):
        """
        SHARIA GUARDRAIL: Only 'buy' side, only 'us_equity'.
        Uses 'notional' for fractional share support.
        """
        if dollar_amount < 1.0:
            log.warning(f"  ⚠️ ${dollar_amount:.2f} too small for {ticker}, skipping")
            return None
        order = {
            "symbol": ticker,
            "notional": round(dollar_amount, 2),
            "side": "buy",          # ☪️ HARDCODED: Long only
            "type": "market",
            "time_in_force": "day",
        }
        log.info(f"  🛒 ORDER BUY {ticker} notional=${dollar_amount:.2f}")
        return self._post(f"{self.base}/v2/orders", order)

    def sell_position(self, ticker):
        """Close entire position for a ticker."""
        log.info(f"  🏷️ ORDER SELL {ticker} (close position)")
        return self._delete(f"{self.base}/v2/positions/{ticker}")

    # ── Prices ────────────────────────────────────────────────────────
    def get_latest_price(self, ticker):
        try:
            r = self._get(f"{self.DATA_URL}/v2/stocks/{ticker}/trades/latest?feed=iex")
            return float(r["trade"]["p"])
        except Exception as e:
            log.warning(f"  ⚠️ Price fetch failed for {ticker}: {e}")
            return None

    def get_bars(self, ticker, start, end, timeframe="1Day"):
        """Fetch OHLCV bars from Alpaca Data API (IEX free feed)."""
        url = (f"{self.DATA_URL}/v2/stocks/{ticker}/bars"
               f"?timeframe={timeframe}&start={start}&end={end}&limit=10000&feed=iex")
        try:
            data = self._get(url)
            bars = data.get("bars", [])
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df = df.rename(columns={"t": "date", "o": "open", "h": "high",
                                    "l": "low", "c": "close", "v": "volume"})
            df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            log.warning(f"  ⚠️ Bars fetch failed for {ticker}: {e}")
            return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# 3. LIVE INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    """
    Loads the trained LightGBM ranker and computes features for today's
    Halal universe. Produces a ranked list of tickers.
    """

    def __init__(self, alpaca: AlpacaClient):
        self.alpaca = alpaca
        self.model = lgb.Booster(model_file=MODEL_PATH)
        with open(FEATURES_PATH) as f:
            self.feature_names = json.load(f)
        log.info(f"  🧠 Model loaded: {len(self.feature_names)} features")

        # Load halal universe
        self.halal_tickers = pd.read_csv(HALAL_CSV)["ticker"].str.upper().tolist()
        log.info(f"  ☪️ Halal universe: {len(self.halal_tickers)} tickers")

        # Load fundamentals for sector/PE/PB features
        self.fundamentals = {}
        if os.path.exists(FUND_CSV):
            fund_df = pd.read_csv(FUND_CSV)
            col = "Ticker" if "Ticker" in fund_df.columns else "ticker"
            for _, row in fund_df.iterrows():
                self.fundamentals[row[col]] = row.to_dict()

        # Load sentiment data for live features
        self.sentiment = {}
        if os.path.exists(SENTIMENT_PATH):
            sent_df = pd.read_parquet(SENTIMENT_PATH)
            sent_df["date"] = pd.to_datetime(sent_df["date"])
            # Keep only the latest row per ticker
            sent_df.sort_values("date", inplace=True)
            for _, row in sent_df.groupby("ticker").tail(1).iterrows():
                self.sentiment[row["ticker"]] = row.to_dict()
            log.info(f"  📰 Sentiment data: {len(self.sentiment)} tickers")
        else:
            log.info("  ⚠️ No sentiment data found (optional)")

    def _build_stock_features(self, df, ticker):
        """Compute per-stock features matching the training pipeline."""
        if len(df) < 252:
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        open_ = df["open"]

        feat = pd.DataFrame(index=df.index)

        # Volatility
        log_ret = np.log(close / close.shift(1))
        feat["vol_20"] = log_ret.rolling(20).std()
        feat["vol_5"]  = log_ret.rolling(5).std()
        feat["vol_ratio"] = feat["vol_5"] / feat["vol_20"].replace(0, np.nan)
        feat["hl_ratio"] = (high - low) / close

        # Returns
        feat["ret_5d"]  = close.pct_change(5)
        feat["ret_20d"] = close.pct_change(20)
        feat["ret_accel_20d"] = feat["ret_20d"] - feat["ret_20d"].shift(20)
        feat["log_return"] = log_ret

        # 52-week proximity
        high_52w = high.rolling(252).max()
        low_52w  = low.rolling(252).min()
        feat["dist_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan)
        feat["dist_52w_low"]  = (close - low_52w) / low_52w.replace(0, np.nan)

        # Dollar volume
        feat["dollar_volume_20d"] = (close * volume).rolling(20).mean()

        # VIX interaction
        feat["vol_x_vix"] = feat["vol_20"]  # placeholder, will be filled with macro

        # Technical indicators via pandas_ta
        feat["RSI_14"] = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None:
            for c in macd.columns:
                feat[c] = macd[c]
        stoch = ta.stoch(high, low, close)
        if stoch is not None:
            for c in stoch.columns:
                feat[c] = stoch[c]
        feat["ADX_14"] = ta.adx(high, low, close, length=14)["ADX_14"] if ta.adx(high, low, close, length=14) is not None else np.nan
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None:
            for c in bb.columns:
                feat[c] = bb[c]
        kc = ta.kc(high, low, close, length=20)
        if kc is not None:
            for c in kc.columns:
                feat[c] = kc[c]
        ema50 = ta.ema(close, length=50)
        feat["EMA_50"] = ema50 / close if ema50 is not None else np.nan
        obv = ta.obv(close, volume)
        if obv is not None:
            feat["OBV"] = obv
        cmf = ta.cmf(high, low, close, volume, length=20)
        if cmf is not None:
            feat["CMF_20"] = cmf
            feat["CMF_delta_5"] = cmf - cmf.shift(5)
        feat["ADXR_14_2"] = ta.adx(high, low, close, length=14)["ADXR_14_2"] if ta.adx(high, low, close, length=14) is not None and "ADXR_14_2" in ta.adx(high, low, close, length=14).columns else np.nan

        # Beta proxy (correlation with SPY is handled via macro merge)
        feat["stock_beta_60d"] = log_ret.rolling(60).std()

        # Fundamentals
        fund = self.fundamentals.get(ticker, {})
        feat["Trailing_PE"]  = fund.get("Trailing_PE", np.nan)
        feat["Price_To_Book"] = fund.get("Price_To_Book", np.nan)
        feat["EV_to_EBITDA"] = fund.get("EV_to_EBITDA", np.nan)

        # Sentiment features
        sent = self.sentiment.get(ticker, {})
        feat["sentiment_mean_3d"]  = sent.get("sentiment_mean_3d", 0.0)
        feat["sentiment_mean_7d"]  = sent.get("sentiment_mean_7d", 0.0)
        feat["sentiment_vol_7d"]   = sent.get("sentiment_vol_7d", 0.0)
        feat["sentiment_momentum"] = sent.get("sentiment_momentum", 0.0)
        feat["news_volume_3d"]     = sent.get("news_volume_3d", 0.0)
        feat["sentiment_std"]      = sent.get("sentiment_std", 0.0)

        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat.ffill(inplace=True)

        # Downcast to float32
        for c in feat.select_dtypes(include=[np.float64]).columns:
            feat[c] = feat[c].astype(np.float32)

        return feat

    def rank_universe(self, exclude_tickers=None):
        """
        Download data, compute features, z-score, predict, rank.
        Returns: list of (ticker, score) sorted descending.
        """
        exclude = exclude_tickers or set()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS + 50)).strftime("%Y-%m-%d")

        all_features = []
        valid_tickers = []

        log.info(f"  📡 Fetching data for {len(self.halal_tickers)} Halal stocks...")
        for ticker in self.halal_tickers:
            if ticker in exclude:
                continue
            try:
                df = self.alpaca.get_bars(ticker, start_date, end_date)
                if df.empty or len(df) < 252:
                    continue
                feat = self._build_stock_features(df, ticker)
                if feat is None or feat.empty:
                    continue
                # Take only the LATEST row
                last_row = feat.iloc[-1:]
                all_features.append(last_row)
                valid_tickers.append(ticker)
            except Exception as e:
                log.debug(f"  Skip {ticker}: {e}")
            gc.collect()

        if not all_features:
            log.warning("  ⚠️ No valid stocks to rank!")
            return []

        log.info(f"  ✅ Features computed for {len(valid_tickers)} stocks")

        # Assemble panel (single date, N stocks)
        panel = pd.concat(all_features, ignore_index=True)
        panel["ticker"] = valid_tickers

        # Cross-sectional z-scoring (today's universe)
        exclude_from_zs = {"ticker", "dollar_volume_20d", "Trailing_PE",
                           "Price_To_Book", "EV_to_EBITDA"}
        for col in panel.select_dtypes(include=[np.number]).columns:
            if col in exclude_from_zs:
                continue
            mean = panel[col].mean()
            std = panel[col].std()
            if std and std > 0:
                panel[f"{col}_xs"] = ((panel[col] - mean) / std).astype(np.float32)

        # Cross-sectional ranks
        for feat, rank_name in [("RSI_14", "RSI_rank"), ("vol_20", "Vol_rank"),
                                ("ret_5d", "Momentum_rank"),
                                ("stock_beta_60d", "Beta_rank"),
                                ("CMF_20", "CMF_rank")]:
            if feat in panel.columns:
                panel[rank_name] = panel[feat].rank(pct=True)

        # Align columns to model's expected features
        for col in self.feature_names:
            if col not in panel.columns:
                panel[col] = 0.0

        X = panel[self.feature_names].fillna(0).values.astype(np.float32)

        # Predict
        scores = self.model.predict(X)

        # Filter out weak sectors (excluded from training)
        results = []
        for ticker, score in zip(valid_tickers, scores):
            fund = self.fundamentals.get(ticker, {})
            sector = fund.get("Sector", "")
            if sector in WEAK_SECTORS:
                continue
            results.append((ticker, score))
        results.sort(key=lambda x: x[1], reverse=True)

        log.info(f"  🏆 Top 5: {[f'{t}({s:.2f})' for t, s in results[:5]]}")
        del panel, all_features
        gc.collect()

        return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN EXECUTION LOOP
# ═══════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 65)
    log.info("🚀 AlphaShariaBot — Live Paper Trading Engine")
    log.info("=" * 65)

    # ── Load API keys ─────────────────────────────────────────────────
    load_dotenv(ENV_PATH)
    api_key = os.getenv("ALPACA_API_KEY")
    secret  = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        log.error("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        log.error("   Add these lines to .env:")
        log.error("   ALPACA_API_KEY=your_key_here")
        log.error("   ALPACA_SECRET_KEY=your_secret_here")
        return

    alpaca = AlpacaClient(api_key, secret)
    wallet = VirtualWallet()

    # ── Idempotency check ─────────────────────────────────────────────
    if wallet.already_ran_today():
        log.info("⏭️  Already ran today. Exiting to prevent double-trading.")
        wallet.summary()
        return

    # ── Market check ──────────────────────────────────────────────────
    try:
        if not alpaca.is_market_open():
            log.info("🔒 Market is closed. Incrementing hold counters only.")
            wallet.increment_holding_days()
            wallet.mark_ran_today()
            wallet.summary()
            return
    except Exception as e:
        log.error(f"❌ Cannot reach Alpaca API: {e}")
        return

    log.info("\n📋 WALLET STATUS (Before)")
    wallet.summary()

    # ── Init Risk Manager ─────────────────────────────────────────────
    risk_mgr = RiskManager()
    equity = wallet.virtual_balance
    risk_mgr.update_circuit_breaker(equity)

    # ══════════════════════════════════════════════════════════════════
    # PHASE A½: STOP-LOSS CHECK (before regular expiration)
    # ══════════════════════════════════════════════════════════════════
    log.info(f"\n{'─'*50}")
    log.info("🛡️ PHASE A½: Stop-loss check...")

    stopped_tickers = risk_mgr.check_stop_losses(
        wallet.positions,
        lambda t: alpaca.get_latest_price(t)
    )
    for ticker in stopped_tickers:
        try:
            price = alpaca.get_latest_price(ticker)
            if price is None:
                continue
            alpaca.sell_position(ticker)
            wallet.close_position(ticker, price)
        except Exception as e:
            log.error(f"  ❌ Stop-loss sell failed for {ticker}: {e}")

    if not stopped_tickers:
        log.info("  No stop-losses triggered.")

    # ══════════════════════════════════════════════════════════════════
    # PHASE A: SELL expired positions (held >= 10 trading days)
    # ══════════════════════════════════════════════════════════════════
    log.info(f"\n{'─'*50}")
    log.info("📤 PHASE A: Checking for expired positions...")

    expired = wallet.expired_positions()
    if expired:
        for pos in expired:
            ticker = pos["ticker"]
            try:
                # Get current price for PnL tracking
                price = alpaca.get_latest_price(ticker)
                if price is None:
                    log.warning(f"  ⚠️ Cannot get price for {ticker}, skipping sell")
                    continue

                # Execute sell on Alpaca
                alpaca.sell_position(ticker)

                # Update virtual wallet
                wallet.close_position(ticker, price)
            except Exception as e:
                log.error(f"  ❌ Failed to sell {ticker}: {e}")
    else:
        log.info("  No positions have reached 10-day holding period.")

    # Increment holding counter for remaining positions
    wallet.increment_holding_days()

    # ══════════════════════════════════════════════════════════════════
    # PHASE B: RANK the Halal universe
    # ══════════════════════════════════════════════════════════════════
    log.info(f"\n{'─'*50}")
    log.info("🧠 PHASE B: Running model inference...")

    # ── Circuit Breaker Check ─────────────────────────────────────────
    updated_equity = wallet.virtual_balance
    can_trade = risk_mgr.update_circuit_breaker(updated_equity)
    if not can_trade:
        log.warning("  🔴 Circuit breaker is active. No new buys.")
        wallet.mark_ran_today()
        risk_mgr.risk_summary(wallet.positions, lambda t: alpaca.get_latest_price(t))
        wallet.summary()
        return

    slots_available = MAX_POSITIONS - wallet.n_open
    if slots_available <= 0:
        log.info(f"  All {MAX_POSITIONS} slots filled. No new buys today.")
        wallet.mark_ran_today()
        wallet.summary()
        return

    engine = InferenceEngine(alpaca)
    rankings = engine.rank_universe(exclude_tickers=wallet.held_tickers())
    del engine; gc.collect()

    if not rankings:
        log.warning("  ⚠️ No ranked stocks available. Skipping buy phase.")
        wallet.mark_ran_today()
        wallet.summary()
        return

    # ══════════════════════════════════════════════════════════════════
    # PHASE C: BUY top-ranked stocks (with risk controls)
    # ══════════════════════════════════════════════════════════════════
    log.info(f"\n{'─'*50}")
    log.info(f"🛒 PHASE C: Buying top {slots_available} picks (risk-managed)...")

    cash = wallet.available_cash_for_new()
    if cash < 1.0:
        log.info(f"  💸 Only ${cash:.2f} cash available. Cannot open new positions.")
        wallet.mark_ran_today()
        wallet.summary()
        return

    # Build candidate list with volatility info for sizing
    candidates = []
    for ticker, score in rankings:
        if len(candidates) >= slots_available * 3:  # check 3x candidates
            break
        try:
            price = alpaca.get_latest_price(ticker)
            if price is None or price <= 0:
                continue

            # Correlation guard: skip if too correlated with holdings
            if not risk_mgr.passes_correlation_check(
                ticker, wallet.held_tickers(), pd.DataFrame(), None
            ):
                continue

            candidates.append({
                "ticker": ticker,
                "score": score,
                "price": price,
                "volatility_20d": 0.02,  # default, would be computed from bars
            })
        except Exception as e:
            log.debug(f"  Skip {ticker}: {e}")

    # Take only top slots_available after filtering
    candidates = candidates[:slots_available]

    if not candidates:
        log.info("  No candidates passed risk filters.")
        wallet.mark_ran_today()
        wallet.summary()
        return

    # Fixed $20 per deal allocation
    bought = 0
    for c in candidates:
        ticker = c["ticker"]
        price = c["price"]
        dollar_amount = min(PER_POSITION, cash)
        if dollar_amount < 1.0:
            log.info(f"  💸 Not enough cash (${cash:.2f}). Stopping buys.")
            break
        try:
            order = alpaca.buy_fractional(ticker, dollar_amount)
            if order is None:
                continue
            shares = round(dollar_amount / price, 6)
            wallet.open_position(ticker, shares, price)
            cash -= dollar_amount
            bought += 1
            log.info(f"  📗 OPEN  {ticker}: {shares} shares @ ${price:.2f} = ${dollar_amount:.2f}")
        except Exception as e:
            log.error(f"  ❌ Failed to buy {ticker}: {e}")

    # ── Wrap up ───────────────────────────────────────────────────────
    wallet.mark_ran_today()
    log.info(f"\n{'─'*50}")
    log.info("📋 WALLET STATUS (After)")
    wallet.summary()

    # Risk health report
    risk_mgr.risk_summary(wallet.positions, lambda t: alpaca.get_latest_price(t))

    # Trade history summary
    history = wallet.state["trade_history"]
    if history:
        wins = sum(1 for t in history if t["pnl"] > 0)
        losses = sum(1 for t in history if t["pnl"] <= 0)
        total_pnl = sum(t["pnl"] for t in history)
        log.info(f"\n📊 TRACK RECORD: {wins}W / {losses}L | Total PnL: ${total_pnl:+.2f}")

    log.info("=" * 65)
    log.info("✅ Daily run complete.")


if __name__ == "__main__":
    main()
