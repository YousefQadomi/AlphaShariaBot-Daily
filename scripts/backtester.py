"""
backtester.py — Vectorized Backtesting Engine for AlphaShariaBot
=================================================================
Replays the trained LightGBM ranker over historical panel data,
simulating the exact strategy from alpha_live.py:
  - Top-K picks (default 5) per day
  - 10-day holding period
  - Equal-weight allocation
  - Halal-only filtering option

Produces: equity curve, drawdown analysis, Sharpe/Sortino ratios,
win rate, alpha vs SPY, per-sector breakdown, and a full trade log.

Usage:
    python scripts/backtester.py
    python scripts/backtester.py --halal-only
    python scripts/backtester.py --top-k 10 --hold-days 5
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import gc
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_manager import RiskManager

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH    = os.path.join(BASE_DIR, "data", "processed", "panel.parquet")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "ranker_model.txt")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "ranker_features.json")
MACRO_DIR     = os.path.join(BASE_DIR, "data", "macro")
OUTPUT_DIR    = os.path.join(BASE_DIR, "data", "backtest")

# ─── Default Strategy Parameters ─────────────────────────────────────────
DEFAULT_TOP_K      = 5
DEFAULT_HOLD_DAYS  = 10
INITIAL_CAPITAL    = 100.0
TRAIN_RATIO        = 0.80


# ═══════════════════════════════════════════════════════════════════════════
# 1. Position Tracker
# ═══════════════════════════════════════════════════════════════════════════
class Position:
    __slots__ = ["ticker", "shares", "entry_price", "entry_date",
                 "entry_cost", "days_held", "sector", "peak_price"]

    def __init__(self, ticker, shares, entry_price, entry_date, sector=""):
        self.ticker = ticker
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.entry_cost = shares * entry_price
        self.days_held = 0
        self.sector = sector
        self.peak_price = entry_price


# ═══════════════════════════════════════════════════════════════════════════
# 2. Backtesting Engine
# ═══════════════════════════════════════════════════════════════════════════
class Backtester:
    def __init__(self, top_k=DEFAULT_TOP_K, hold_days=DEFAULT_HOLD_DAYS,
                 halal_only=False, initial_capital=INITIAL_CAPITAL):
        self.top_k = top_k
        self.hold_days = hold_days
        self.halal_only = halal_only
        self.initial_capital = initial_capital

        # State
        self.cash = initial_capital
        self.positions = []        # list of Position
        self.trade_log = []        # completed trades
        self.equity_curve = []     # (date, equity)
        self.daily_returns = []

    def _load_model(self):
        print("🧠 Loading LightGBM ranker model...")
        self.model = lgb.Booster(model_file=MODEL_PATH)
        with open(FEATURES_PATH) as f:
            self.feature_names = json.load(f)
        print(f"   ✅ Model loaded: {len(self.feature_names)} features, "
              f"{self.model.num_trees()} trees")

    def _load_panel(self):
        print("📦 Loading panel data...")
        panel = pd.read_parquet(PANEL_PATH)
        if "date" not in panel.columns:
            panel = panel.reset_index()
        panel["date"] = pd.to_datetime(panel["date"])
        panel.sort_values(["date", "ticker"], inplace=True)
        panel.reset_index(drop=True, inplace=True)

        # Force float32
        for col in panel.select_dtypes(include=[np.float64]).columns:
            panel[col] = panel[col].astype(np.float32)

        return panel

    def _load_spy_benchmark(self):
        """Load SPY for benchmark comparison."""
        spy_path = os.path.join(MACRO_DIR, "SPY.parquet")
        if not os.path.exists(spy_path):
            return None
        spy = pd.read_parquet(spy_path)
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        spy = spy[["close"]].rename(columns={"close": "spy_close"})
        return spy

    def _get_price(self, panel_day, ticker):
        """Get close price for a ticker on a given day."""
        row = panel_day[panel_day["ticker"] == ticker]
        if row.empty:
            return None
        return float(row["close"].iloc[0])

    def _portfolio_value(self, panel_day):
        """Compute total portfolio value (cash + positions at current prices)."""
        value = self.cash
        for pos in self.positions:
            price = self._get_price(panel_day, pos.ticker)
            if price is not None:
                value += pos.shares * price
            else:
                value += pos.entry_cost  # fallback
        return value

    def run(self):
        """Run the full backtest."""
        self._load_model()
        panel = self._load_panel()
        spy_data = self._load_spy_benchmark()

        # ── Split: only test on unseen data ───────────────────────────
        unique_dates = sorted(panel["date"].unique())
        split_idx = int(len(unique_dates) * TRAIN_RATIO)
        test_dates = unique_dates[split_idx:]

        print(f"\n📅 Backtest Period:")
        print(f"   Train: {unique_dates[0].date()} → "
              f"{unique_dates[split_idx-1].date()} ({split_idx} days)")
        print(f"   Test:  {test_dates[0].date()} → "
              f"{test_dates[-1].date()} ({len(test_dates)} days)")
        print(f"\n⚙️  Strategy: Top-{self.top_k}, "
              f"{self.hold_days}d hold, "
              f"{'Halal-only' if self.halal_only else 'Full universe'}")
        print(f"   Initial capital: ${self.initial_capital:.2f}\n")

        # Ensure feature columns exist
        for col in self.feature_names:
            if col not in panel.columns:
                panel[col] = 0.0

        prev_equity = self.initial_capital

        # Init risk manager for backtest
        risk_mgr = RiskManager()
        risk_mgr.peak_equity = self.initial_capital
        stop_loss_sells = 0
        correlation_blocks = 0
        circuit_breaker_days = 0

        # ── Day-by-day simulation ─────────────────────────────────────
        for day_idx, date in enumerate(test_dates):
            panel_day = panel[panel["date"] == date]
            if panel_day.empty:
                continue

            # --- STOP-LOSS CHECK (new: before expiry check) ---
            for pos in list(self.positions):
                price = self._get_price(panel_day, pos.ticker)
                if price is None:
                    continue
                # Update peak price tracking
                pos.peak_price = max(pos.peak_price, price)
                stop_pct = risk_mgr.config["stop_loss_pct"]
                drawdown_from_peak = (price - pos.peak_price) / pos.peak_price
                if drawdown_from_peak <= stop_pct:
                    exit_value = pos.shares * price
                    pnl = exit_value - pos.entry_cost
                    self.cash += exit_value
                    self.trade_log.append({
                        "ticker": pos.ticker,
                        "entry_date": str(pos.entry_date.date()),
                        "exit_date": str(date.date()),
                        "entry_price": round(pos.entry_price, 4),
                        "exit_price": round(price, 4),
                        "shares": round(pos.shares, 6),
                        "pnl": round(pnl, 4),
                        "return_pct": round(pnl / pos.entry_cost * 100, 2),
                        "days_held": pos.days_held,
                        "sector": pos.sector,
                        "exit_reason": "stop_loss",
                    })
                    self.positions.remove(pos)
                    stop_loss_sells += 1

            # --- SELL expired positions ---
            expired = [p for p in self.positions if p.days_held >= self.hold_days]
            for pos in expired:
                exit_price = self._get_price(panel_day, pos.ticker)
                if exit_price is None:
                    continue
                exit_value = pos.shares * exit_price
                pnl = exit_value - pos.entry_cost
                self.cash += exit_value
                self.trade_log.append({
                    "ticker": pos.ticker,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(date.date()),
                    "entry_price": round(pos.entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "shares": round(pos.shares, 6),
                    "pnl": round(pnl, 4),
                    "return_pct": round(pnl / pos.entry_cost * 100, 2),
                    "days_held": pos.days_held,
                    "sector": pos.sector,
                    "exit_reason": "expiry",
                })
                self.positions.remove(pos)

            # Increment hold counter
            for pos in self.positions:
                pos.days_held += 1

            # --- CIRCUIT BREAKER CHECK ---
            equity_now = self._portfolio_value(panel_day)
            can_trade = risk_mgr.update_circuit_breaker(equity_now)

            # --- RANK universe ---
            slots = self.top_k - len(self.positions)
            if slots > 0 and self.cash > 1.0 and can_trade:
                universe = panel_day.copy()
                if self.halal_only:
                    universe = universe[universe["is_halal"] == 1]

                # Exclude already-held tickers
                held = {p.ticker for p in self.positions}
                universe = universe[~universe["ticker"].isin(held)]

                if len(universe) >= 5:
                    X = universe[self.feature_names].fillna(0).values
                    X = X.astype(np.float32)
                    scores = self.model.predict(X)
                    universe = universe.copy()
                    universe["score"] = scores
                    ranked = universe.nlargest(slots * 3, "score")

                    # --- FILTER by correlation guard ---
                    filtered_picks = []
                    for _, row in ranked.iterrows():
                        if len(filtered_picks) >= slots:
                            break
                        ticker = row["ticker"]
                        if risk_mgr.passes_correlation_check(
                            ticker, held | {p["ticker"] for p in filtered_picks},
                            panel_day, None
                        ):
                            filtered_picks.append(row.to_dict())
                        else:
                            correlation_blocks += 1

                    # --- BUY with vol-weighted sizing ---
                    if filtered_picks:
                        candidates = []
                        for pick in filtered_picks:
                            vol = pick.get("vol_20", 0.02)
                            if vol <= 0:
                                vol = 0.02
                            candidates.append({
                                "ticker": pick["ticker"],
                                "volatility_20d": vol,
                            })
                        allocations = risk_mgr.inverse_vol_weights(
                            candidates, self.cash
                        )
                        for pick in filtered_picks:
                            ticker = pick["ticker"]
                            price = float(pick["close"])
                            if price <= 0:
                                continue
                            dollar_amount = min(
                                allocations.get(ticker, 0), self.cash
                            )
                            if dollar_amount < 0.50:
                                continue
                            shares = dollar_amount / price
                            sector = pick.get("Sector", "")
                            self.cash -= dollar_amount
                            self.positions.append(Position(
                                ticker, shares, price, date, sector
                            ))
            elif not can_trade:
                circuit_breaker_days += 1

            # --- Track equity ---
            equity = self._portfolio_value(panel_day)
            self.equity_curve.append({
                "date": date,
                "equity": round(equity, 4),
                "cash": round(self.cash, 4),
                "n_positions": len(self.positions),
            })

            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_ret)
            prev_equity = equity

            # Progress
            if (day_idx + 1) % 50 == 0:
                print(f"   Day {day_idx+1}/{len(test_dates)} | "
                      f"Equity: ${equity:.2f} | "
                      f"Positions: {len(self.positions)}")

        # ── Final forced liquidation ──────────────────────────────────
        last_day = panel[panel["date"] == test_dates[-1]]
        for pos in list(self.positions):
            price = self._get_price(last_day, pos.ticker)
            if price:
                exit_value = pos.shares * price
                pnl = exit_value - pos.entry_cost
                self.cash += exit_value
                self.trade_log.append({
                    "ticker": pos.ticker,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(test_dates[-1].date()),
                    "entry_price": round(pos.entry_price, 4),
                    "exit_price": round(price, 4),
                    "shares": round(pos.shares, 6),
                    "pnl": round(pnl, 4),
                    "return_pct": round(pnl / pos.entry_cost * 100, 2),
                    "days_held": pos.days_held,
                    "sector": pos.get("sector", "") if hasattr(pos, "sector") else "",
                })
        self.positions.clear()

        # ── Generate report ───────────────────────────────────────────
        risk_stats = {
            "stop_loss_sells": stop_loss_sells,
            "correlation_blocks": correlation_blocks,
            "circuit_breaker_days": circuit_breaker_days,
        }
        self._generate_report(spy_data, test_dates, risk_stats)

        del panel
        gc.collect()

    def _generate_report(self, spy_data, test_dates, risk_stats=None):
        """Generate full backtest report with metrics and charts."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        eq_df = pd.DataFrame(self.equity_curve)
        eq_df.set_index("date", inplace=True)

        trades = pd.DataFrame(self.trade_log)
        final_equity = eq_df["equity"].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        # ── Core Metrics ──────────────────────────────────────────────
        returns = np.array(self.daily_returns)
        n_days = len(returns)
        ann_factor = 252

        avg_daily = np.mean(returns)
        std_daily = np.std(returns) if np.std(returns) > 0 else 1e-9
        sharpe = (avg_daily / std_daily) * np.sqrt(ann_factor)

        downside = returns[returns < 0]
        down_std = np.std(downside) if len(downside) > 0 and np.std(downside) > 0 else 1e-9
        sortino = (avg_daily / down_std) * np.sqrt(ann_factor)

        # Max drawdown
        cum_max = eq_df["equity"].cummax()
        drawdown = (eq_df["equity"] - cum_max) / cum_max
        max_dd = drawdown.min() * 100
        max_dd_date = drawdown.idxmin()

        # Trade stats
        if not trades.empty:
            wins = (trades["pnl"] > 0).sum()
            losses = (trades["pnl"] <= 0).sum()
            total_trades = len(trades)
            win_rate = wins / total_trades * 100
            avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
            avg_loss = trades[trades["pnl"] <= 0]["pnl"].mean() if losses > 0 else 0
            profit_factor = abs(avg_win * wins) / abs(avg_loss * losses) if losses > 0 and avg_loss != 0 else float('inf')
            avg_return_pct = trades["return_pct"].mean()
        else:
            wins = losses = total_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = avg_return_pct = 0

        # ── SPY Benchmark ─────────────────────────────────────────────
        spy_return = None
        spy_sharpe = None
        if spy_data is not None:
            spy_test = spy_data.loc[
                (spy_data.index >= test_dates[0]) & (spy_data.index <= test_dates[-1])
            ]
            if len(spy_test) > 1:
                spy_return = (spy_test["spy_close"].iloc[-1] / spy_test["spy_close"].iloc[0] - 1) * 100
                spy_daily = spy_test["spy_close"].pct_change().dropna()
                spy_sharpe_val = (spy_daily.mean() / spy_daily.std()) * np.sqrt(252) if spy_daily.std() > 0 else 0
                spy_sharpe = spy_sharpe_val

        # ── Print Report ──────────────────────────────────────────────
        print(f"\n{'='*65}")
        print(f"📊 BACKTEST RESULTS — AlphaShariaBot")
        print(f"{'='*65}")
        print(f"  Strategy:        Top-{self.top_k}, {self.hold_days}d hold, "
              f"{'Halal-only' if self.halal_only else 'Full universe'}")
        print(f"  Period:          {test_dates[0].date()} → {test_dates[-1].date()} ({n_days} days)")
        print(f"  Initial Capital: ${self.initial_capital:.2f}")
        print(f"  Final Equity:    ${final_equity:.2f}")
        print(f"")
        print(f"{'─'*40}")
        print(f"  📈 RETURNS")
        print(f"  Total Return:    {total_return:+.2f}%")
        print(f"  Annualized:      {((1+total_return/100)**(252/max(n_days,1))-1)*100:+.2f}%")
        if spy_return is not None:
            print(f"  SPY Return:      {spy_return:+.2f}%")
            print(f"  Alpha vs SPY:    {total_return - spy_return:+.2f}%")
        print(f"")
        print(f"{'─'*40}")
        print(f"  📉 RISK")
        print(f"  Sharpe Ratio:    {sharpe:.3f}")
        print(f"  Sortino Ratio:   {sortino:.3f}")
        if spy_sharpe is not None:
            print(f"  SPY Sharpe:      {spy_sharpe:.3f}")
        print(f"  Max Drawdown:    {max_dd:.2f}%")
        print(f"  Max DD Date:     {max_dd_date.date()}")
        print(f"  Daily Volatility:{std_daily*100:.3f}%")
        print(f"")
        print(f"{'─'*40}")
        print(f"  🎯 TRADES")
        print(f"  Total Trades:    {total_trades}")
        print(f"  Win Rate:        {win_rate:.1f}%")
        print(f"  Avg Win:         ${avg_win:.4f}")
        print(f"  Avg Loss:        ${avg_loss:.4f}")
        print(f"  Profit Factor:   {profit_factor:.2f}")
        print(f"  Avg Return/Trade:{avg_return_pct:.2f}%")

        # Risk manager stats
        if risk_stats:
            print(f"")
            print(f"{'─'*40}")
            print(f"  🛡️ RISK MANAGEMENT")
            print(f"  Stop-loss sells:     {risk_stats.get('stop_loss_sells', 0)}")
            print(f"  Correlation blocks:  {risk_stats.get('correlation_blocks', 0)}")
            print(f"  Circuit breaker days:{risk_stats.get('circuit_breaker_days', 0)}")

        # Sector breakdown
        if not trades.empty and "sector" in trades.columns:
            sec_stats = trades.groupby("sector").agg(
                trades=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                avg_ret=("return_pct", "mean"),
                win_rate=("pnl", lambda x: (x > 0).mean() * 100),
            ).sort_values("trades", ascending=False)

            if len(sec_stats) > 0:
                print(f"\n{'─'*40}")
                print(f"  📊 SECTOR BREAKDOWN")
                for sec, row in sec_stats.iterrows():
                    if not sec:
                        continue
                    print(f"  {sec:20s} | {int(row['trades']):4d} trades | "
                          f"PnL: ${row['total_pnl']:+.3f} | "
                          f"WR: {row['win_rate']:.0f}%")

        print(f"{'='*65}")

        # ── Save Trade Log ────────────────────────────────────────────
        if not trades.empty:
            trades.to_csv(os.path.join(OUTPUT_DIR, "trade_log.csv"), index=False)

        # ── Save Metrics JSON ─────────────────────────────────────────
        metrics = {
            "strategy": f"Top-{self.top_k}_{self.hold_days}d",
            "halal_only": self.halal_only,
            "period_start": str(test_dates[0].date()),
            "period_end": str(test_dates[-1].date()),
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 4),
            "total_return_pct": round(total_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4),
            "spy_return_pct": round(spy_return, 4) if spy_return else None,
            "alpha_vs_spy": round(total_return - spy_return, 4) if spy_return else None,
        }
        with open(os.path.join(OUTPUT_DIR, "backtest_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # ── Plot Equity Curve ─────────────────────────────────────────
        self._plot_equity(eq_df, spy_data, test_dates, drawdown, metrics)

    def _plot_equity(self, eq_df, spy_data, test_dates, drawdown, metrics):
        """Generate professional equity curve chart."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.suptitle("AlphaShariaBot — Backtest Results", fontsize=16, fontweight="bold")

        # --- Equity Curve ---
        ax1 = axes[0]
        norm_eq = eq_df["equity"] / eq_df["equity"].iloc[0] * 100
        ax1.plot(eq_df.index, norm_eq, color="#2196F3", linewidth=1.5, label="Strategy")

        if spy_data is not None:
            spy_test = spy_data.loc[
                (spy_data.index >= test_dates[0]) & (spy_data.index <= test_dates[-1])
            ]
            if len(spy_test) > 1:
                norm_spy = spy_test["spy_close"] / spy_test["spy_close"].iloc[0] * 100
                ax1.plot(spy_test.index, norm_spy, color="#FF9800", linewidth=1.2,
                         alpha=0.8, label="SPY (Buy & Hold)")

        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                      f"Return: {metrics['total_return_pct']:+.1f}% | "
                      f"MaxDD: {metrics['max_drawdown_pct']:.1f}%")

        # --- Drawdown ---
        ax2 = axes[1]
        ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                         color="#F44336", alpha=0.4)
        ax2.set_ylabel("Drawdown %")
        ax2.grid(True, alpha=0.3)

        # --- Number of positions ---
        ax3 = axes[2]
        ax3.bar(eq_df.index, eq_df["n_positions"], color="#4CAF50", alpha=0.6, width=1.5)
        ax3.set_ylabel("Positions")
        ax3.set_xlabel("Date")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n📊 Charts → {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AlphaShariaBot Backtester")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of top picks per day (default: {DEFAULT_TOP_K})")
    parser.add_argument("--hold-days", type=int, default=DEFAULT_HOLD_DAYS,
                        help=f"Holding period in trading days (default: {DEFAULT_HOLD_DAYS})")
    parser.add_argument("--halal-only", action="store_true",
                        help="Restrict to Halal-compliant stocks only")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help=f"Initial capital (default: ${INITIAL_CAPITAL})")
    args = parser.parse_args()

    # Validate
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Run 'python scripts/train_ranker.py' first.")
        return
    if not os.path.exists(PANEL_PATH):
        print(f"❌ Panel not found at {PANEL_PATH}")
        print("   Run 'python scripts/feature_engineering.py' first.")
        return

    print("🚀 AlphaShariaBot — Backtesting Engine\n")

    bt = Backtester(
        top_k=args.top_k,
        hold_days=args.hold_days,
        halal_only=args.halal_only,
        initial_capital=args.capital,
    )
    bt.run()
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()
