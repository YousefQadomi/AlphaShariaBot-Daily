"""
risk_manager.py — Dynamic Risk Management for AlphaShariaBot
==============================================================
This module provides four layers of capital protection that the
original system completely lacked:

  1. TRAILING STOP-LOSS
     If a position drops X% from its highest price since entry,
     sell it immediately. Default: -8%.
     WHY: The old system held for exactly 10 days regardless.
     A stock could crash -30% and the bot would just watch.

  2. VOLATILITY-BASED POSITION SIZING (Inverse Vol Weighting)
     Instead of splitting $20 equally across 5 stocks, allocate
     MORE money to calm/stable stocks and LESS to wild/volatile ones.
     WHY: Equal-weighting means a high-volatility stock has
     disproportionate impact on portfolio returns.

  3. CORRELATION GUARD
     Before buying a new stock, check its return correlation with
     stocks already in the portfolio. If correlation > threshold,
     skip it and pick the next-ranked stock instead.
     WHY: If all 5 picks are tech stocks that move together,
     diversification is an illusion — one bad day wipes all 5.

  4. MAX DRAWDOWN CIRCUIT BREAKER
     If the portfolio drops X% from its all-time high, STOP
     opening new positions until recovery. Existing positions are
     still managed (stop-losses still fire, expirations still sell).
     WHY: Protects against regime changes (market crashes, black
     swans) where the model's predictions become unreliable.

Usage:
    # Standalone (used as a module by alpha_live.py and backtester.py)
    from risk_manager import RiskManager
    rm = RiskManager()
    rm.check_stop_losses(positions, get_price_fn)
    weights = rm.inverse_vol_weights(candidates, volatilities)
    ok = rm.correlation_check(new_ticker, held_tickers, panel_day)
    can_trade = rm.circuit_breaker_ok(current_equity, peak_equity)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK_CFG     = os.path.join(BASE_DIR, "data", "live", "risk_config.json")
RISK_LOG     = os.path.join(BASE_DIR, "logs", "risk_events.log")

log = logging.getLogger("RiskManager")


# ═══════════════════════════════════════════════════════════════════════════
# Default Risk Parameters (Swing Trading — 10-day holds)
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    # ── Stop-Loss ─────────────────────────────────────────────────────
    # If a position falls X% from its peak price since entry → sell.
    # -8% is standard for medium-frequency US equity strategies.
    "stop_loss_pct": -0.08,

    # ── Volatility Sizing ─────────────────────────────────────────────
    # Target annualized volatility per position.
    # If a stock has 40% ann. vol and target is 20%, it gets half-weight.
    "target_vol_annual": 0.20,
    # Floor/cap on individual position weight (% of available cash)
    "min_weight": 0.10,  # no position less than 10% of available cash
    "max_weight": 0.40,  # no position more than 40% of available cash

    # ── Correlation Guard ─────────────────────────────────────────────
    # Max correlation allowed between a new pick and any existing position.
    # 0.70 = moderate correlation. Pairs above this are too similar.
    "max_correlation": 0.70,
    # Lookback period (trading days) for correlation computation
    "corr_lookback": 60,

    # ── Circuit Breaker ───────────────────────────────────────────────
    # If portfolio drops X% from its all-time peak → stop new buys.
    "max_drawdown_pct": -0.15,   # -15% drawdown = freeze new buys
    # Resume trading when drawdown recovers above this level
    "resume_drawdown_pct": -0.10,  # resume at -10%
}


# ═══════════════════════════════════════════════════════════════════════════
# Intraday Risk Parameters (Day Trading — same-day close)
# ═══════════════════════════════════════════════════════════════════════════
INTRADAY_CONFIG = {
    # ── ATR-Based Dynamic Stop-Loss ──────────────────────────────────
    "stop_loss_atr_multiplier": 2.0,    # stop = entry - (multiplier * ATR)
    "stop_loss_min_pct": -0.025,         # floor: never wider than -2.5%
    "stop_loss_max_pct": -0.005,         # ceiling: never tighter than -0.5%
    "stop_loss_fallback_pct": -0.018,    # used when ATR unavailable (-1.8%)

    # ── Legacy stop (kept for config completeness) ───────────────────
    "stop_loss_pct": -0.018,            # matches fallback; used by base class

    # ── Trailing Stop ────────────────────────────────────────────────
    "trailing_stop_pct": -0.008,        # -0.8% trailing after activation
    "trailing_activation_pct": 0.012,   # activate trailing after +1.2%

    # ── Scaling Exits ────────────────────────────────────────────────
    "partial_exit_pct": 0.015,           # +1.5% → sell 50%
    "partial_trailing_stop_pct": -0.005, # tighter trailing for remainder
    "full_take_profit_pct": 0.050,       # +5.0% → sell everything
    "take_profit_pct": 0.050,            # alias for full_take_profit_pct

    # ── Time-Based Stop ──────────────────────────────────────────────
    "time_stop_minutes": 45,
    "time_stop_min_gain_pct": 0.004,

    # ── Daily Risk Limits ─────────────────────────────────────────────
    "max_daily_loss_pct": -0.03,        # -3% daily loss → halt trading
    "max_daily_trades": 30,             # cap trades per day
    "max_consecutive_losses": 3,        # 3 losses in a row → pause

    # ── Position Sizing ───────────────────────────────────────────────
    "risk_per_trade_pct": 0.03,         # risk 3% of equity per trade
    "max_position_pct": 0.40,           # max 40% of equity in one stock
    "max_sector_exposure": 0.40,        # max 40% in one sector

    # ── Volatility Sizing ─────────────────────────────────────────────
    "target_vol_annual": 0.20,
    "min_weight": 0.05,
    "max_weight": 0.25,

    # ── Correlation Guard ─────────────────────────────────────────────
    "max_correlation": 0.70,
    "corr_lookback": 60,

    # ── Circuit Breaker (tighter for intraday) ────────────────────────
    "max_drawdown_pct": -0.05,          # -5% drawdown = freeze
    "resume_drawdown_pct": -0.03,

    # ── Spread & Cost Guard ───────────────────────────────────────────
    "max_entry_spread_pct": 0.001,      # skip if spread > 0.1%
    "min_dollar_volume_1h": 100_000,    # min $ volume in last hour
    "min_profit_after_costs": 0.002,    # min expected profit after costs
}


# ═══════════════════════════════════════════════════════════════════════════
# RiskManager Class
# ═══════════════════════════════════════════════════════════════════════════
class RiskManager:
    """
    Pluggable risk management module. Can be used by:
      - alpha_live.py (live paper trading)
      - backtester.py (historical simulation)

    All methods are stateless (except circuit breaker tracking).
    Configuration can be customized via risk_config.json.
    """

    def __init__(self, config=None):
        self.config = config or self._load_config()
        self.peak_equity = 0.0  # track peak for drawdown calc
        self.circuit_breaker_active = False
        log.info(f"  🛡️ Risk Manager initialized")
        log.info(f"     Stop-loss:    {self.config['stop_loss_pct']*100:.0f}%")
        log.info(f"     Max corr:     {self.config['max_correlation']:.2f}")
        log.info(f"     Circuit break: {self.config['max_drawdown_pct']*100:.0f}% DD")

    def _load_config(self):
        """Load config from JSON, or use defaults."""
        if os.path.exists(RISK_CFG):
            with open(RISK_CFG) as f:
                user_cfg = json.load(f)
            # Merge with defaults (user overrides take precedence)
            cfg = {**DEFAULT_CONFIG, **user_cfg}
            return cfg
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        """Save current config to disk for future reference."""
        os.makedirs(os.path.dirname(RISK_CFG), exist_ok=True)
        with open(RISK_CFG, "w") as f:
            json.dump(self.config, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════
    # 1. TRAILING STOP-LOSS
    # ═══════════════════════════════════════════════════════════════════
    def check_stop_losses(self, positions, get_price_fn):
        """
        Check each position against its trailing stop-loss.

        Args:
            positions: list of position dicts with 'ticker', 'entry_price',
                       and optionally 'peak_price'
            get_price_fn: callable(ticker) → current_price or None

        Returns:
            list of tickers that should be sold (stop-loss triggered)

        HOW IT WORKS:
        ─────────────
        For each position, we track the highest price since entry (peak).
        If the current price is (stop_loss_pct)% below that peak, we
        trigger a sell. This is called a "trailing" stop-loss because the
        threshold moves UP as the stock rises, but never moves DOWN.

        Example with -8% stop:
          Buy at $100 → peak becomes $100 → stop at $92
          Stock rises to $110 → peak now $110 → stop rises to $101.20
          Stock falls to $101 → no trigger (above $101.20)
          Stock falls to $100 → TRIGGERED (below $101.20)
          Result: Locked in ~$1 loss instead of waiting for 10-day exit
        """
        stop_pct = self.config["stop_loss_pct"]
        triggered = []

        for pos in positions:
            ticker = pos["ticker"]
            current_price = get_price_fn(ticker)
            if current_price is None:
                continue

            entry_price = pos["entry_price"]
            # Track peak (the highest price this position has ever seen)
            peak = pos.get("peak_price", entry_price)
            peak = max(peak, current_price)
            pos["peak_price"] = peak  # update for persistence

            # Compute trailing drawdown from peak
            drawdown_from_peak = (current_price - peak) / peak

            if drawdown_from_peak <= stop_pct:
                pnl_pct = (current_price - entry_price) / entry_price * 100
                log.warning(
                    f"  🛑 STOP-LOSS {ticker}: "
                    f"price=${current_price:.2f} is {drawdown_from_peak*100:.1f}% "
                    f"below peak ${peak:.2f} | PnL: {pnl_pct:+.1f}%"
                )
                triggered.append(ticker)

        return triggered

    # ═══════════════════════════════════════════════════════════════════
    # 2. VOLATILITY-BASED POSITION SIZING
    # ═══════════════════════════════════════════════════════════════════
    def inverse_vol_weights(self, candidates, cash_available):
        """
        Compute dollar allocation per stock based on inverse volatility.

        Args:
            candidates: list of dicts with 'ticker' and 'volatility_20d'
                        (20-day realized volatility of daily returns)
            cash_available: total cash to allocate

        Returns:
            dict of {ticker: dollar_amount}

        HOW IT WORKS:
        ─────────────
        Equal-weight: $100 / 5 stocks = $20 each
        The problem: Stock A (tech) has 3x the volatility of Stock B (utility).
        $20 in A contributes 3x more portfolio risk than $20 in B.

        Inverse-vol: weight ∝ 1/volatility
          Stock A (vol=0.30): weight = 1/0.30 = 3.33
          Stock B (vol=0.10): weight = 1/0.10 = 10.0
          Normalized: A gets $25, B gets $75 (of $100 for 2 stocks)

        This way, each position contributes EQUAL RISK to the portfolio,
        not equal dollars. It's the same principle used by risk-parity
        hedge funds like Bridgewater.
        """
        target_vol = self.config["target_vol_annual"]
        min_w = self.config["min_weight"]
        max_w = self.config["max_weight"]

        if not candidates:
            return {}

        # Compute inverse-vol weights
        raw_weights = {}
        for c in candidates:
            ticker = c["ticker"]
            vol = c.get("volatility_20d", target_vol)
            if vol <= 0:
                vol = target_vol  # fallback for zero/negative vol
            # Annualize the 20-day vol (vol * sqrt(252))
            ann_vol = vol * np.sqrt(252)
            # Weight proportional to target_vol / actual_vol
            raw_weights[ticker] = target_vol / max(ann_vol, 0.01)

        # Normalize to sum to 1.0, then apply floor/cap
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            # Fallback to equal weight
            n = len(candidates)
            return {c["ticker"]: cash_available / n for c in candidates}

        weights = {}
        for ticker, w in raw_weights.items():
            normalized = w / total_weight
            clamped = max(min_w, min(max_w, normalized))
            weights[ticker] = clamped

        # Re-normalize after clamping to sum to 1.0
        weight_sum = sum(weights.values())
        allocations = {}
        for ticker, w in weights.items():
            dollar = round(cash_available * (w / weight_sum), 2)
            allocations[ticker] = max(1.0, dollar)  # minimum $1 per position

        log.info(f"  ⚖️ Vol-weighted allocations:")
        for ticker, dollar in allocations.items():
            vol = next((c.get("volatility_20d", 0) for c in candidates
                       if c["ticker"] == ticker), 0)
            log.info(f"     {ticker}: ${dollar:.2f} (vol={vol:.4f})")

        return allocations

    # ═══════════════════════════════════════════════════════════════════
    # 3. CORRELATION GUARD
    # ═══════════════════════════════════════════════════════════════════
    def passes_correlation_check(self, new_ticker, held_tickers, panel_day,
                                  full_panel=None):
        """
        Check if a new ticker is too correlated with existing holdings.

        Args:
            new_ticker:   ticker we want to buy
            held_tickers: set of tickers currently in the portfolio
            panel_day:    DataFrame for today's date (used for feature-based corr)
            full_panel:   full historical panel (optional, for return-based corr)

        Returns:
            True if the ticker passes (low correlation), False if blocked.

        HOW IT WORKS:
        ─────────────
        If you hold AAPL, MSFT, GOOG, and NVDA — all four are tech stocks
        that move together. Adding AMZN (also tech) gives you 5 positions
        but NOT 5 independent bets. One bad tech day and all 5 drop.

        This check computes the pairwise return correlation between the
        candidate stock and each held stock. If ANY pair exceeds the
        threshold (default 0.70), the candidate is rejected.

        We use the Sector column as a fast heuristic: if the new stock is
        in the same sector as 2+ existing positions, that's already
        suspicious.
        """
        max_corr = self.config["max_correlation"]

        if not held_tickers:
            return True  # nothing to compare against

        # ── Fast heuristic: sector concentration check ────────────────
        if "Sector" in panel_day.columns:
            new_row = panel_day[panel_day["ticker"] == new_ticker]
            if not new_row.empty:
                new_sector = new_row["Sector"].iloc[0]
                if new_sector and new_sector != "Unknown":
                    held_rows = panel_day[panel_day["ticker"].isin(held_tickers)]
                    same_sector = (held_rows["Sector"] == new_sector).sum()
                    # Block if already 2+ positions in same sector
                    if same_sector >= 2:
                        log.info(f"     🚫 {new_ticker}: blocked by sector guard "
                                f"({same_sector} positions in {new_sector})")
                        return False

        # ── Return-based correlation (if full panel available) ────────
        if full_panel is not None and "close" in full_panel.columns:
            lookback = self.config["corr_lookback"]
            all_tickers = list(held_tickers) + [new_ticker]

            # Get recent close prices for all relevant tickers
            recent = full_panel[
                full_panel["ticker"].isin(all_tickers)
            ].copy()

            if len(recent) > 0 and "date" in recent.columns:
                # Pivot to get price series per ticker
                pivot = recent.pivot_table(
                    index="date", columns="ticker", values="close"
                )
                # Keep only the last N days
                if len(pivot) > lookback:
                    pivot = pivot.tail(lookback)

                # Compute returns
                returns = pivot.pct_change().dropna()

                if new_ticker in returns.columns:
                    for held in held_tickers:
                        if held in returns.columns:
                            corr = returns[new_ticker].corr(returns[held])
                            if not np.isnan(corr) and abs(corr) > max_corr:
                                log.info(
                                    f"     🚫 {new_ticker}: blocked by "
                                    f"correlation guard (ρ={corr:.2f} "
                                    f"with {held}, limit={max_corr})"
                                )
                                return False

        return True

    # ═══════════════════════════════════════════════════════════════════
    # 4. CIRCUIT BREAKER (Max Drawdown)
    # ═══════════════════════════════════════════════════════════════════
    def update_circuit_breaker(self, current_equity):
        """
        Track portfolio peak and determine if new buys should be blocked.

        Args:
            current_equity: total portfolio value right now

        Returns:
            True if trading is allowed, False if circuit breaker is active.

        HOW IT WORKS:
        ─────────────
        We track the highest portfolio value ever seen (the "high water mark").
        If the current value drops below (high_water * (1 + max_drawdown_pct)),
        we activate the circuit breaker:

          Peak: $120, max_drawdown: -15%
          Threshold: $120 * 0.85 = $102
          If equity drops to $100 → CIRCUIT BREAKER ON → no new buys
          If equity recovers to $108 ($120 * 0.90) → RESUME trading

        This prevents the bot from compounding losses during market crashes
        or periods when the model's predictions become unreliable.
        """
        max_dd = self.config["max_drawdown_pct"]
        resume_dd = self.config["resume_drawdown_pct"]

        # Update peak
        self.peak_equity = max(self.peak_equity, current_equity)

        if self.peak_equity <= 0:
            return True

        # Current drawdown
        drawdown = (current_equity - self.peak_equity) / self.peak_equity

        if self.circuit_breaker_active:
            # Check if we've recovered enough to resume
            if drawdown >= resume_dd:
                self.circuit_breaker_active = False
                log.info(
                    f"  🟢 CIRCUIT BREAKER OFF: drawdown recovered to "
                    f"{drawdown*100:.1f}% (resume threshold: {resume_dd*100:.0f}%)"
                )
                return True
            else:
                log.warning(
                    f"  🔴 CIRCUIT BREAKER ACTIVE: drawdown={drawdown*100:.1f}% "
                    f"(need {resume_dd*100:.0f}% to resume)"
                )
                return False
        else:
            # Check if drawdown exceeds threshold
            if drawdown <= max_dd:
                self.circuit_breaker_active = True
                log.warning(
                    f"  🔴 CIRCUIT BREAKER TRIGGERED: drawdown={drawdown*100:.1f}% "
                    f"exceeded limit of {max_dd*100:.0f}% | "
                    f"Peak=${self.peak_equity:.2f} → Now=${current_equity:.2f}"
                )
                return False
            return True

    # ═══════════════════════════════════════════════════════════════════
    # Risk Summary
    # ═══════════════════════════════════════════════════════════════════
    def risk_summary(self, positions, get_price_fn):
        """Print a risk health report for current positions."""
        if not positions:
            log.info("  🛡️ No positions — nothing to report.")
            return

        log.info(f"\n  🛡️ RISK HEALTH CHECK")
        log.info(f"  {'─'*45}")

        total_invested = 0
        total_unrealized = 0
        max_pos_loss = 0
        stop_pct = self.config["stop_loss_pct"]

        for pos in positions:
            ticker = pos["ticker"]
            price = get_price_fn(ticker)
            if price is None:
                continue

            entry = pos["entry_price"]
            peak = pos.get("peak_price", entry)
            cost = pos.get("entry_cost", pos["shares"] * entry)
            current_val = pos["shares"] * price
            pnl_pct = (price - entry) / entry * 100
            dd_from_peak = (price - peak) / peak * 100

            total_invested += cost
            total_unrealized += (current_val - cost)
            max_pos_loss = min(max_pos_loss, pnl_pct)

            # Stop distance
            stop_price = peak * (1 + stop_pct)
            stop_dist = (price - stop_price) / price * 100

            status = "🟢" if pnl_pct >= 0 else ("🟡" if pnl_pct > stop_pct*100 else "🔴")
            log.info(
                f"  {status} {ticker:6s} | PnL:{pnl_pct:+6.1f}% | "
                f"Peak DD:{dd_from_peak:+5.1f}% | Stop dist:{stop_dist:+5.1f}%"
            )

        drawdown = ((self.peak_equity - (self.peak_equity + total_unrealized))
                    / self.peak_equity * 100) if self.peak_equity > 0 else 0

        log.info(f"  {'─'*45}")
        log.info(f"  Portfolio DD from peak: {drawdown:+.1f}%")
        log.info(f"  Worst position:        {max_pos_loss:+.1f}%")
        breaker = "🔴 ACTIVE" if self.circuit_breaker_active else "🟢 OK"
        log.info(f"  Circuit breaker:       {breaker}")


# ═══════════════════════════════════════════════════════════════════════════
# IntradayRiskManager — Day Trading Risk Controls
# ═══════════════════════════════════════════════════════════════════════════
class IntradayRiskManager(RiskManager):
    """
    Extended risk manager for intraday (day) trading.

    Adds on top of the base RiskManager:
      - ATR-based dynamic stop-loss (adapts to each stock's volatility)
      - Scaling exits: partial exit at +0.8%, full take-profit at +2.0%
      - Time-based stop: close stale positions after 90 minutes
      - Smart position sizing for small accounts (<$500)
      - Position reconciliation (wallet vs Alpaca)
      - Per-trade take-profit and trailing stop-loss
      - Daily loss limit (halt trading if exceeded)
      - Daily trade counter (prevent overtrading)
      - Consecutive loss tracker (pause after streak)
      - Spread / cost guard (skip expensive entries)
      - Risk-based position sizing (1% risk per trade)
    """

    def __init__(self, config=None):
        intraday_cfg = config or INTRADAY_CONFIG.copy()
        super().__init__(config=intraday_cfg)

        # Daily tracking (reset each morning)
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.daily_halted = False
        self.start_of_day_equity = 0.0

        log.info(f"  ⚡ Intraday Risk Manager initialized")
        log.info(f"     ATR multiplier: {self.config.get('stop_loss_atr_multiplier', 1.5)}x")
        log.info(f"     Stop fallback:  {self.config.get('stop_loss_fallback_pct', -0.012)*100:.1f}%")
        log.info(f"     Partial exit:   +{self.config.get('partial_exit_pct', 0.008)*100:.1f}%")
        log.info(f"     Full TP:        +{self.config.get('full_take_profit_pct', 0.020)*100:.1f}%")
        log.info(f"     Time stop:      {self.config.get('time_stop_minutes', 90)} min")
        log.info(f"     Daily loss cap: {self.config['max_daily_loss_pct']*100:.1f}%")
        log.info(f"     Max trades/day: {self.config['max_daily_trades']}")

        # Save config on init so the Gradio dashboard can read it
        self.save_config()

    def reset_daily(self, equity):
        """Call at market open to reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.daily_halted = False
        self.start_of_day_equity = equity
        log.info(f"  🌅 Daily risk counters reset. SOD equity: ${equity:.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # ATR-Based Dynamic Stop-Loss Calculation
    # ═══════════════════════════════════════════════════════════════════
    def _compute_atr_stop_pct(self, entry_price, atr_value):
        """
        Compute stop-loss percentage from ATR for a given stock.

        stop_price = entry_price - (multiplier × ATR)
        stop_pct   = (stop_price - entry_price) / entry_price
                   = -(multiplier × ATR) / entry_price

        The result is clamped between stop_loss_min_pct (-2.5%)
        and stop_loss_max_pct (-0.5%) to prevent absurd values.

        Args:
            entry_price: price at which the position was opened
            atr_value:   Average True Range value for the stock

        Returns:
            float: negative stop-loss percentage (e.g. -0.015 for -1.5%)
        """
        multiplier = self.config.get("stop_loss_atr_multiplier", 1.5)
        min_pct = self.config.get("stop_loss_min_pct", -0.025)   # floor: -2.5%
        max_pct = self.config.get("stop_loss_max_pct", -0.005)   # ceiling: -0.5%

        if entry_price <= 0 or atr_value <= 0:
            return self.config.get("stop_loss_fallback_pct", -0.012)

        raw_stop_pct = -(multiplier * atr_value) / entry_price

        # Clamp: min_pct is more negative (wider), max_pct is less negative (tighter)
        # e.g. clamp between -0.025 and -0.005
        clamped = max(min_pct, min(max_pct, raw_stop_pct))
        return clamped

    # ═══════════════════════════════════════════════════════════════════
    # Per-Trade Exit Signals (ATR stops + Scaling exits + Time stop)
    # ═══════════════════════════════════════════════════════════════════
    def check_intraday_exits(self, positions, get_price_fn, atr_values=None):
        """
        Check all positions for intraday exit signals with ATR-based
        dynamic stops, scaling exits, and time-based stops.

        Exit types returned:
          - 'stop_loss'     : price fell below ATR-based (or fallback) stop
          - 'partial_exit'  : price gained +0.8% → caller should sell 50%
          - 'trailing_stop' : remainder dropped from peak after partial exit
          - 'take_profit'   : full take-profit at +2.0%
          - 'time_stop'     : position stale for 90+ min with <0.3% gain

        Args:
            positions:    list of position dicts with 'ticker', 'entry_price',
                          and optionally 'peak_price', 'entry_time',
                          'partial_exited' (bool)
            get_price_fn: callable(ticker) → current_price or None
            atr_values:   optional dict {ticker: ATR_value} for dynamic stops

        Returns:
            list of (ticker, exit_reason) that should be closed.
        """
        atr_values = atr_values or {}

        # Config values
        fallback_stop = self.config.get("stop_loss_fallback_pct", -0.012)
        trail_pct = self.config.get("trailing_stop_pct", -0.005)
        trail_act = self.config.get("trailing_activation_pct", 0.005)
        partial_exit_pct = self.config.get("partial_exit_pct", 0.008)
        partial_trail_pct = self.config.get("partial_trailing_stop_pct", -0.003)
        full_tp_pct = self.config.get("full_take_profit_pct", 0.020)
        time_stop_min = self.config.get("time_stop_minutes", 90)
        time_stop_gain = self.config.get("time_stop_min_gain_pct", 0.003)

        exits = []
        now = datetime.now(ZoneInfo("America/New_York"))

        for pos in positions:
            ticker = pos["ticker"]
            current_price = get_price_fn(ticker)
            if current_price is None:
                continue

            entry_price = pos["entry_price"]
            pnl_pct = (current_price - entry_price) / entry_price

            # Update peak price
            peak = pos.get("peak_price", entry_price)
            peak = max(peak, current_price)
            pos["peak_price"] = peak

            already_partial = pos.get("partial_exited", False)

            # ── 1. ATR-based dynamic stop-loss ────────────────────────
            if ticker in atr_values and atr_values[ticker] > 0:
                sl_pct = self._compute_atr_stop_pct(entry_price, atr_values[ticker])
                log.debug(
                    f"  📐 {ticker}: ATR stop = {sl_pct*100:+.2f}% "
                    f"(ATR={atr_values[ticker]:.4f}, entry=${entry_price:.2f})"
                )
            else:
                sl_pct = fallback_stop
                log.debug(
                    f"  📐 {ticker}: Using fallback stop = {sl_pct*100:+.2f}% "
                    f"(no ATR available)"
                )

            if pnl_pct <= sl_pct:
                log.warning(
                    f"  🛑 STOP-LOSS {ticker}: {pnl_pct*100:+.2f}% "
                    f"(dynamic limit: {sl_pct*100:.2f}%)"
                )
                exits.append((ticker, "stop_loss"))
                continue

            # ── 2. Full take-profit at +2.0% ─────────────────────────
            if pnl_pct >= full_tp_pct:
                log.info(
                    f"  🎯 FULL TAKE-PROFIT {ticker}: {pnl_pct*100:+.2f}% "
                    f"(target: +{full_tp_pct*100:.1f}%)"
                )
                exits.append((ticker, "take_profit"))
                continue

            # ── 3. Scaling exit: partial at +0.8% ────────────────────
            if not already_partial and pnl_pct >= partial_exit_pct:
                log.info(
                    f"  📊 PARTIAL EXIT {ticker}: {pnl_pct*100:+.2f}% "
                    f"(threshold: +{partial_exit_pct*100:.1f}%) — sell 50%"
                )
                pos["partial_exited"] = True
                exits.append((ticker, "partial_exit"))
                continue

            # ── 4. Tighter trailing stop after partial exit ──────────
            if already_partial:
                dd_from_peak = (current_price - peak) / peak
                if dd_from_peak <= partial_trail_pct:
                    log.info(
                        f"  📉 PARTIAL TRAILING STOP {ticker}: "
                        f"peak gain → now {pnl_pct*100:+.2f}%, "
                        f"dd from peak {dd_from_peak*100:+.2f}% "
                        f"(limit: {partial_trail_pct*100:.1f}%)"
                    )
                    exits.append((ticker, "trailing_stop"))
                    continue

            # ── 5. Normal trailing stop (before partial exit) ────────
            if not already_partial:
                gain_from_entry = (peak - entry_price) / entry_price
                if gain_from_entry >= trail_act:
                    dd_from_peak = (current_price - peak) / peak
                    if dd_from_peak <= trail_pct:
                        log.info(
                            f"  📉 TRAILING STOP {ticker}: "
                            f"peak gain {gain_from_entry*100:+.1f}% → "
                            f"now {pnl_pct*100:+.1f}%"
                        )
                        exits.append((ticker, "trailing_stop"))
                        continue

            # ── 6. Time-based stop ───────────────────────────────────
            entry_time = pos.get("entry_time")
            if entry_time is not None:
                # Parse entry_time if it's a string
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except (ValueError, TypeError):
                        log.debug(f"  ⏰ {ticker}: Could not parse entry_time '{entry_time}'")
                        continue

                # Ensure timezone-aware
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=ZoneInfo("America/New_York"))

                elapsed_minutes = (now - entry_time).total_seconds() / 60.0

                if elapsed_minutes >= time_stop_min and pnl_pct < time_stop_gain:
                    log.info(
                        f"  ⏰ TIME STOP {ticker}: {elapsed_minutes:.0f} min elapsed, "
                        f"gain {pnl_pct*100:+.2f}% < +{time_stop_gain*100:.1f}% threshold"
                    )
                    exits.append((ticker, "time_stop"))
                    continue

        return exits

    # ═══════════════════════════════════════════════════════════════════
    # Daily Risk Limits
    # ═══════════════════════════════════════════════════════════════════
    def can_open_new_trade(self):
        """
        Check if we are allowed to open a new position.
        Returns (allowed: bool, reason: str).
        """
        if self.daily_halted:
            return False, "Daily halt active (loss limit reached)"

        max_trades = self.config["max_daily_trades"]
        if self.daily_trades >= max_trades:
            return False, f"Max daily trades reached ({max_trades})"

        max_losses = self.config["max_consecutive_losses"]
        if self.consecutive_losses >= max_losses:
            return False, f"Consecutive loss limit ({max_losses})"

        if self.circuit_breaker_active:
            return False, "Circuit breaker active"

        return True, "OK"

    def record_trade_result(self, pnl, equity):
        """
        Call after each closed trade to update daily counters.
        Returns True if trading should continue, False if halted.
        """
        self.daily_pnl += pnl
        self.daily_trades += 1

        if pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check daily loss limit
        if self.start_of_day_equity > 0:
            daily_return = self.daily_pnl / self.start_of_day_equity
            max_loss = self.config["max_daily_loss_pct"]
            if daily_return <= max_loss:
                self.daily_halted = True
                log.warning(
                    f"  🚨 DAILY LOSS LIMIT HIT: {daily_return*100:+.1f}% "
                    f"(limit: {max_loss*100:.1f}%) — halting all new trades"
                )
                return False

        # Update circuit breaker
        self.update_circuit_breaker(equity)
        return not self.daily_halted

    # ═══════════════════════════════════════════════════════════════════
    # Cost-Aware Position Sizing (Original)
    # ═══════════════════════════════════════════════════════════════════
    def calculate_position_size(self, equity, entry_price, stop_price,
                                spread_pct=0.0003):
        """
        Risk-based position sizing: risk 1% of equity per trade.

        Accounts for:
          - Distance to stop-loss
          - Round-trip spread cost
          - Slippage estimate (0.02%)

        Returns: dollar amount to invest (capped at max_position_pct).
        """
        risk_pct = self.config["risk_per_trade_pct"]
        max_pos = self.config["max_position_pct"]

        risk_amount = equity * risk_pct  # e.g., $1000 * 1% = $10
        risk_per_share = abs(entry_price - stop_price)
        spread_cost = entry_price * spread_pct * 2  # round trip
        slippage = entry_price * 0.0002  # 0.02% estimate
        total_risk_per_share = risk_per_share + spread_cost + slippage

        if total_risk_per_share <= 0:
            return 0.0

        shares = risk_amount / total_risk_per_share
        dollar_amount = shares * entry_price

        # Cap at max position size
        max_dollar = equity * max_pos
        dollar_amount = min(dollar_amount, max_dollar)

        # Floor at $1 (Alpaca minimum)
        return max(1.0, round(dollar_amount, 2))

    # ═══════════════════════════════════════════════════════════════════
    # Smart Position Sizing v2 (Account-Size Aware + ATR)
    # ═══════════════════════════════════════════════════════════════════
    def calculate_position_size_v2(self, equity, entry_price, atr, spread_pct=0.0003):
        """
        Smart position sizing based on account size and ATR.

        Adapts risk parameters and max positions to account size:
          - Small  (<$500):   2% risk per trade, max 5 positions
          - Medium ($500–$5k): 1.5% risk per trade, max 8 positions
          - Large  (>$5k):    1% risk per trade, max 12 positions

        Uses ATR to compute stop distance, then sizes the position so that
        hitting the stop would cost exactly (risk_pct × equity).

        Also enforces a $15 minimum position size (so profits remain
        meaningful after spread costs on small accounts).

        Args:
            equity:     current portfolio equity
            entry_price: expected entry price of the stock
            atr:        Average True Range value for the stock
            spread_pct: estimated bid-ask spread as a fraction (default 0.03%)

        Returns:
            dict with keys:
              - 'dollar_amount': float, how much to invest
              - 'max_positions': int, max concurrent positions for this equity
              - 'risk_pct': float, risk percentage used
              - 'stop_pct': float, the ATR-based stop percentage applied
        """
        # ── Tier the account ─────────────────────────────────────────
        if equity < 500:
            risk_pct = 0.03     # 3% risk for small accounts
            max_positions = 3
            tier = "small"
        elif equity < 5000:
            risk_pct = 0.025    # 2.5% risk for medium accounts
            max_positions = 3
            tier = "medium"
        else:
            risk_pct = 0.015    # 1.5% risk for large accounts
            max_positions = 5
            tier = "large"

        log.info(
            f"  💰 Position sizing v2: equity=${equity:.2f} → {tier} account "
            f"(risk={risk_pct*100:.1f}%, max_pos={max_positions})"
        )

        if entry_price <= 0:
            log.warning("  ⚠️ Invalid entry_price for position sizing")
            return {
                "dollar_amount": 0.0,
                "max_positions": max_positions,
                "risk_pct": risk_pct,
                "stop_pct": 0.0,
            }

        # ── Compute ATR-based stop distance ──────────────────────────
        if atr > 0:
            stop_pct = self._compute_atr_stop_pct(entry_price, atr)
        else:
            stop_pct = self.config.get("stop_loss_fallback_pct", -0.012)

        stop_distance = abs(stop_pct) * entry_price  # dollar distance to stop

        # ── Account for costs ────────────────────────────────────────
        spread_cost = entry_price * spread_pct * 2   # round-trip spread
        slippage = entry_price * 0.0002              # 0.02% slippage estimate
        total_risk_per_share = stop_distance + spread_cost + slippage

        if total_risk_per_share <= 0:
            log.warning(f"  ⚠️ Zero risk per share — cannot size position")
            return {
                "dollar_amount": 0.0,
                "max_positions": max_positions,
                "risk_pct": risk_pct,
                "stop_pct": stop_pct,
            }

        # ── Size the position ────────────────────────────────────────
        risk_amount = equity * risk_pct              # e.g., $100 * 2% = $2
        shares = risk_amount / total_risk_per_share
        dollar_amount = shares * entry_price

        # Cap at (equity / max_positions) * 1.5 so we can concentrate more
        max_dollar_per_pos = (equity / max_positions) * 1.5
        dollar_amount = min(dollar_amount, max_dollar_per_pos)

        # Also cap at the general max_position_pct
        max_pos_pct = self.config.get("max_position_pct", 0.15)
        dollar_amount = min(dollar_amount, equity * max_pos_pct)

        # Enforce $15 minimum for small accounts (to make profits meaningful)
        min_position = 50.0
        if dollar_amount < min_position:
            if equity >= min_position:
                dollar_amount = min_position
                log.info(
                    f"  📏 Position size floored to ${min_position:.0f} minimum"
                )
            else:
                # Account too small even for minimum — use whatever we have
                dollar_amount = max(1.0, equity * 0.5)
                log.warning(
                    f"  ⚠️ Account too small for ${min_position:.0f} minimum, "
                    f"using ${dollar_amount:.2f}"
                )

        dollar_amount = round(dollar_amount, 2)

        log.info(
            f"  📏 Position size: ${dollar_amount:.2f} "
            f"(risk=${risk_amount:.2f}, stop={stop_pct*100:+.2f}%, "
            f"ATR=${atr:.4f})"
        )

        return {
            "dollar_amount": dollar_amount,
            "max_positions": max_positions,
            "risk_pct": risk_pct,
            "stop_pct": stop_pct,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Position Reconciliation (Wallet vs Alpaca)
    # ═══════════════════════════════════════════════════════════════════
    def reconcile_positions(self, wallet_positions, alpaca_positions):
        """
        Compare wallet state with actual Alpaca positions.

        This catches dangerous discrepancies that can happen when:
          - A sell order filled but the wallet wasn't updated
          - The bot crashed mid-trade
          - Manual trades were placed on Alpaca
          - Network errors caused missed order confirmations

        Args:
            wallet_positions: list of dicts from the bot's wallet, each with
                              at least {'ticker': str, 'shares': float}
            alpaca_positions: list of dicts from Alpaca API, each with
                              at least {'ticker': str, 'shares': float}
                              (or 'qty' as an alias for 'shares')

        Returns:
            dict with keys:
              - 'orphaned_wallet': list of tickers in wallet but NOT in Alpaca
                                   (phantom positions — we think we own them
                                    but Alpaca says we don't)
              - 'orphaned_alpaca': list of tickers in Alpaca but NOT in wallet
                                   (ghost positions — Alpaca has them but our
                                    bot doesn't know about them)
              - 'quantity_mismatch': list of dicts with
                                     {'ticker', 'wallet_shares', 'alpaca_shares'}
                                     where both sides have the position but
                                     the share count differs
              - 'matched': list of tickers that match perfectly
        """
        # Build lookup dicts: ticker → shares
        wallet_map = {}
        for pos in wallet_positions:
            ticker = pos.get("ticker", pos.get("symbol", ""))
            shares = float(pos.get("shares", pos.get("qty", 0)))
            if ticker:
                wallet_map[ticker] = shares

        alpaca_map = {}
        for pos in alpaca_positions:
            ticker = pos.get("ticker", pos.get("symbol", ""))
            shares = float(pos.get("shares", pos.get("qty", 0)))
            if ticker:
                alpaca_map[ticker] = shares

        wallet_tickers = set(wallet_map.keys())
        alpaca_tickers = set(alpaca_map.keys())

        # Tickers only in wallet (phantom)
        orphaned_wallet = sorted(wallet_tickers - alpaca_tickers)
        # Tickers only in Alpaca (ghost)
        orphaned_alpaca = sorted(alpaca_tickers - wallet_tickers)
        # Tickers in both
        common = wallet_tickers & alpaca_tickers

        matched = []
        quantity_mismatch = []

        for ticker in sorted(common):
            w_shares = wallet_map[ticker]
            a_shares = alpaca_map[ticker]
            # Allow tiny float differences (< 0.001 share)
            if abs(w_shares - a_shares) < 0.001:
                matched.append(ticker)
            else:
                quantity_mismatch.append({
                    "ticker": ticker,
                    "wallet_shares": w_shares,
                    "alpaca_shares": a_shares,
                })

        # Log results
        if orphaned_wallet:
            log.warning(
                f"  ⚠️ RECONCILIATION: {len(orphaned_wallet)} phantom positions "
                f"(in wallet, not in Alpaca): {orphaned_wallet}"
            )
        if orphaned_alpaca:
            log.warning(
                f"  ⚠️ RECONCILIATION: {len(orphaned_alpaca)} ghost positions "
                f"(in Alpaca, not in wallet): {orphaned_alpaca}"
            )
        if quantity_mismatch:
            for m in quantity_mismatch:
                log.warning(
                    f"  ⚠️ RECONCILIATION: {m['ticker']} qty mismatch — "
                    f"wallet={m['wallet_shares']:.4f}, "
                    f"alpaca={m['alpaca_shares']:.4f}"
                )
        if not orphaned_wallet and not orphaned_alpaca and not quantity_mismatch:
            log.info(
                f"  ✅ RECONCILIATION: All {len(matched)} positions match perfectly"
            )

        return {
            "orphaned_wallet": orphaned_wallet,
            "orphaned_alpaca": orphaned_alpaca,
            "quantity_mismatch": quantity_mismatch,
            "matched": matched,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Spread & Cost Guard
    # ═══════════════════════════════════════════════════════════════════
    def passes_spread_check(self, bid, ask, price):
        """
        Check if a stock's bid-ask spread is acceptable for entry.
        Wide spreads eat into profits on intraday trades.
        """
        if price <= 0:
            return False
        spread = (ask - bid) / price
        max_spread = self.config.get("max_entry_spread_pct", 0.001)
        if spread > max_spread:
            log.debug(f"  Spread too wide: {spread*100:.3f}% > {max_spread*100:.1f}%")
            return False
        return True

    def is_trade_profitable(self, expected_return_pct, spread_pct):
        """
        Verify expected profit exceeds transaction costs.
        Prevents entering trades where fees eat all profit.
        """
        total_cost = (spread_pct * 2) + 0.0002 + 0.0005  # spread + slip + safety
        min_profit = self.config.get("min_profit_after_costs", 0.002)
        return expected_return_pct > max(total_cost, min_profit)

    def intraday_summary(self):
        """Print intraday risk health report."""
        log.info(f"\n  ⚡ INTRADAY RISK SUMMARY")
        log.info(f"  {'─'*45}")
        daily_ret = (self.daily_pnl / self.start_of_day_equity * 100
                     if self.start_of_day_equity > 0 else 0)
        log.info(f"  Daily P&L:         ${self.daily_pnl:+.2f} ({daily_ret:+.1f}%)")
        log.info(f"  Trades today:      {self.daily_trades}")
        log.info(f"  Consecutive losses: {self.consecutive_losses}")
        halted = "🔴 HALTED" if self.daily_halted else "🟢 ACTIVE"
        log.info(f"  Trading status:    {halted}")
        breaker = "🔴 ACTIVE" if self.circuit_breaker_active else "🟢 OK"
        log.info(f"  Circuit breaker:   {breaker}")


if __name__ == "__main__":
    # Quick test / config generation
    rm = RiskManager()
    rm.save_config()
    print(f"\n✅ Risk config saved to {RISK_CFG}")
    print(f"   Edit this file to customize risk parameters.")
    print(f"\n📋 Current settings:")
    for k, v in rm.config.items():
        print(f"   {k}: {v}")

    print(f"\n📋 Intraday config:")
    for k, v in INTRADAY_CONFIG.items():
        print(f"   {k}: {v}")
