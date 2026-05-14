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

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK_CFG     = os.path.join(BASE_DIR, "data", "live", "risk_config.json")
RISK_LOG     = os.path.join(BASE_DIR, "logs", "risk_events.log")

log = logging.getLogger("RiskManager")


# ═══════════════════════════════════════════════════════════════════════════
# Default Risk Parameters
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


if __name__ == "__main__":
    # Quick test / config generation
    rm = RiskManager()
    rm.save_config()
    print(f"\n✅ Risk config saved to {RISK_CFG}")
    print(f"   Edit this file to customize risk parameters.")
    print(f"\n📋 Current settings:")
    for k, v in rm.config.items():
        print(f"   {k}: {v}")
