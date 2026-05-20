"""
app.py — AlphaShariaBot Cloud Scheduler + Dashboard (Day Trading V2)
=====================================================================
Runs on Hugging Face Spaces (Gradio).
- Intraday trading: Adaptive scan intervals (60s opening, 5min normal, 10min midday)
- News polling:     Every 15 min during market hours
- Force-close:      At 3:50 PM ET
- Dashboard:        Shows live wallet, trades, risk health, intraday stats
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import threading
import time
import subprocess
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import gradio as gr

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WALLET_PATH = os.path.join(BASE_DIR, "data", "live", "intraday_wallet.json")
RISK_CFG    = os.path.join(BASE_DIR, "data", "live", "risk_config.json")
LOG_PATH    = os.path.join(BASE_DIR, "logs", "alpha_intraday.log")
SCHED_LOG   = os.path.join(BASE_DIR, "logs", "scheduler.log")

# ─── Logging ──────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCHEDULER] %(message)s",
    handlers=[
        logging.FileHandler(SCHED_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
slog = logging.getLogger("scheduler")

# ─── Helper: Run a script and capture output ─────────────────────────────
def run_script(script_name, extra_args=None):
    """Run a Python script and return (success, output)."""
    script_path = os.path.join(BASE_DIR, "scripts", script_name)
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
    slog.info(f"▶ Starting {script_name} {extra_args or ''}...")
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            slog.info(f"✅ {script_name} completed successfully.")
        else:
            slog.error(f"❌ {script_name} failed (exit {result.returncode})")
        return result.returncode == 0, output[-2000:]
    except subprocess.TimeoutExpired:
        slog.error(f"⏰ {script_name} timed out.")
        return False, "Script timed out."
    except Exception as e:
        slog.error(f"💥 {script_name} error: {e}")
        return False, str(e)

# ─── Load wallet data for dashboard ──────────────────────────────────────
def load_wallet():
    """Load and return intraday wallet state as formatted text."""
    try:
        with open(WALLET_PATH, "r") as f:
            w = json.load(f)
    except FileNotFoundError:
        return "No wallet file found. Run the bot first."

    lines = []
    lines.append(f"⚡ Mode: INTRADAY DAY TRADING")
    lines.append(f"💰 Initial Balance:  ${w.get('initial_balance', 0):,.2f}")
    lines.append(f"💵 Cash Available:   ${w.get('cash', 0):,.2f}")
    lines.append(f"📊 Realized PnL:     ${w.get('realized_pnl', 0):+,.2f}")
    lines.append(f"📂 Open Positions:   {len(w.get('positions', []))}")
    lines.append(f"📜 Total Trades:     {len(w.get('trade_history', []))}")
    lines.append(f"📅 Last Run:         {w.get('last_run_date', 'Never')}")

    # Daily stats (intraday mode)
    daily = w.get("daily_stats", [])
    if daily:
        last_day = daily[-1]
        lines.append(f"\n{'─'*50}")
        lines.append("TODAY'S STATS:")
        lines.append(f"  Date:    {last_day.get('date', '?')}")
        lines.append(f"  PnL:     ${last_day.get('pnl', 0):+.2f}")
        lines.append(f"  Trades:  {last_day.get('trades', 0)}")
        lines.append(f"  Equity:  ${last_day.get('equity', 0):.2f}")

    # Positions table
    positions = w.get("positions", [])
    if positions:
        lines.append(f"\n{'─'*50}")
        lines.append("OPEN POSITIONS:")
        lines.append(f"{'Ticker':<8} {'Shares':>10} {'Entry $':>10} {'Score':>6}")
        lines.append(f"{'─'*8} {'─'*10} {'─'*10} {'─'*6}")
        for p in positions[:20]:
            score = p.get("entry_score", 0)
            lines.append(f"{p['ticker']:<8} {p['shares']:>10.4f} "
                        f"${p['entry_price']:>9.2f} {score:>6.1f}")

    # Recent trades
    history = w.get("trade_history", [])
    if history:
        lines.append(f"\n{'─'*50}")
        lines.append("RECENT TRADES (last 15):")
        lines.append(f"{'Ticker':<8} {'PnL':>10} {'Return':>8} {'Reason':>12}")
        lines.append(f"{'─'*8} {'─'*10} {'─'*8} {'─'*12}")
        for t in history[-15:]:
            pnl_str = f"${t['pnl']:+.2f}"
            ret_str = f"{t.get('return_pct', 0):+.1f}%"
            reason = t.get("exit_reason", "?")[:12]
            lines.append(f"{t['ticker']:<8} {pnl_str:>10} {ret_str:>8} {reason:>12}")

    # Win rate
    if history:
        wins = sum(1 for t in history if t["pnl"] > 0)
        total = len(history)
        avg_pnl = sum(t["pnl"] for t in history) / total
        lines.append(f"\n{'─'*50}")
        lines.append(f"📊 Win Rate: {wins}/{total} ({wins/total*100:.0f}%) | "
                     f"Avg PnL: ${avg_pnl:.4f}")

    return "\n".join(lines)

def load_risk_config():
    """Load risk config as formatted text."""
    try:
        with open(RISK_CFG, "r") as f:
            cfg = json.load(f)
        lines = ["⚡ INTRADAY RISK CONFIG:"]
        lines.append(f"  Stop-loss:        {cfg.get('stop_loss_pct', -0.008) * 100:.1f}%")
        lines.append(f"  Take-profit:      {cfg.get('take_profit_pct', 0.015) * 100:.1f}%")
        lines.append(f"  Trailing stop:    {cfg.get('trailing_stop_pct', -0.005) * 100:.1f}%")
        lines.append(f"  Daily loss limit: {cfg.get('max_daily_loss_pct', -0.03) * 100:.1f}%")
        lines.append(f"  Max trades/day:   {cfg.get('max_daily_trades', 30)}")
        lines.append(f"  Max consec loss:  {cfg.get('max_consecutive_losses', 5)}")
        lines.append(f"  Max spread:       {cfg.get('max_entry_spread_pct', 0.001) * 100:.2f}%")
        lines.append(f"  Circuit breaker:  {cfg.get('max_drawdown_pct', -0.05) * 100:.1f}% DD")
        return "\n".join(lines)
    except FileNotFoundError:
        return ("No risk config found. Using defaults:\n"
                "  Stop-loss: -0.8% | Take-profit: +1.5% | Daily limit: -3%")

def load_logs():
    """Load last 50 lines of the intraday log."""
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-50:])
    except FileNotFoundError:
        return "No log file found. Run the bot first."

# ─── Scheduler State ─────────────────────────────────────────────────────
scheduler_status = {
    "mode": "INTRADAY",
    "scan_last_run": "Never",
    "scan_next_run": "Starting...",
    "news_last_run": "Never",
    "scans_today": 0,
    "trades_today": 0,
    "last_output": "",
    "running": True,
}

def get_scheduler_status():
    """Return scheduler status as formatted text."""
    s = scheduler_status
    et_now = datetime.now(ZoneInfo("America/New_York"))
    is_market_hours = (
        et_now.weekday() < 5 and
        (et_now.hour > 9 or (et_now.hour == 9 and et_now.minute >= 30)) and
        et_now.hour < 16
    )

    lines = [
        f"⚡ Mode:                 {s['mode']} DAY TRADING",
        f"🕐 Current Time (ET):     {et_now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"📡 Market Hours:          {'YES ✅' if is_market_hours else 'NO 🔒'}",
        f"🔍 Last Scan:             {s['scan_last_run']}",
        f"🔍 Next Scan:             {s['scan_next_run']}",
        f"📰 News Last Poll:        {s['news_last_run']}",
        f"📊 Scans Today:           {s['scans_today']}",
        f"🔄 Scheduler Active:      {'YES' if s['running'] else 'NO'}",
    ]
    return "\n".join(lines)


# ─── Adaptive Scan Interval ──────────────────────────────────────────────
def _get_adaptive_interval():
    """Return scan interval based on market session phase."""
    now = datetime.now(ZoneInfo("America/New_York"))
    market_open = now.replace(hour=9, minute=30, second=0)
    mins = max(0, (now - market_open).total_seconds() / 60)
    if mins < 30:
        return 60, "opening"      # first 30 min: every 60s
    elif mins >= 330:
        return 120, "power_hour"  # 3:00 PM+: every 2 min
    elif 120 < mins < 270:
        return 600, "midday"      # 11:30-2:00: every 10 min
    else:
        return 300, "normal"      # default: every 5 min


# ─── Background Scheduler ────────────────────────────────────────────────
def scheduler_loop():
    """
    Background thread for intraday trading V2:
    - Adaptive scan intervals (60s opening, 5min normal, 10min midday, 2min power hour)
    - News polling every 15 minutes
    - Force-closes all at 3:50 PM ET
    """
    last_scan_time = 0
    last_news_time = 0
    today_date = None

    slog.info("⚡ Intraday scheduler V2 started (adaptive intervals).")

    while scheduler_status["running"]:
        try:
            now = datetime.now(ZoneInfo("America/New_York"))
            now_ts = time.time()

            # Reset daily counters
            if today_date != now.date():
                today_date = now.date()
                scheduler_status["scans_today"] = 0
                scheduler_status["trades_today"] = 0

            is_weekday = now.weekday() < 5
            is_market_hours = (
                (now.hour > 9 or (now.hour == 9 and now.minute >= 35)) and
                (now.hour < 15 or (now.hour == 15 and now.minute <= 55))
            )

            if is_weekday and is_market_hours:
                scan_interval, phase = _get_adaptive_interval()

                # ── Trading scan (adaptive interval) ──────────────────
                if now_ts - last_scan_time >= scan_interval:
                    ok, output = run_script("alpha_intraday.py")
                    last_scan_time = now_ts
                    scheduler_status["scan_last_run"] = now.strftime(
                        "%H:%M:%S ET")
                    next_scan = now + timedelta(seconds=scan_interval)
                    scheduler_status["scan_next_run"] = (
                        f"{next_scan.strftime('%H:%M:%S ET')} ({phase})"
                    )
                    scheduler_status["scans_today"] += 1
                    scheduler_status["last_output"] = f"[Scan] {output[-500:]}"

                # ── News poll every 15 minutes ────────────────────────
                if now_ts - last_news_time >= 900:
                    last_news_time = now_ts
                    scheduler_status["news_last_run"] = now.strftime(
                        "%H:%M:%S ET")

                # ── Force-close at 3:50 PM ────────────────────────────
                if now.hour == 15 and now.minute >= 50:
                    slog.info("🔔 EOD Force-close triggered")
                    run_script("alpha_intraday.py", ["--force-close"])

            # ── Weekly halal screener refresh (Sunday midnight or Monday pre-market)
            if now.weekday() == 0 and now.hour == 7 and now.minute < 2:
                slog.info("☪️ Weekly halal universe refresh...")
                run_script("sharia_screener.py")

            # Check less frequently outside market hours
            sleep_time = 15 if is_market_hours else 60
            time.sleep(sleep_time)

        except Exception as e:
            slog.error(f"Scheduler error: {e}")
            time.sleep(60)


# ─── Manual Trigger Functions ────────────────────────────────────────────
def manual_run_scan():
    """Manually trigger one intraday scan."""
    ok, output = run_script("alpha_intraday.py")
    et_now = datetime.now(ZoneInfo("America/New_York"))
    scheduler_status["scan_last_run"] = et_now.strftime("%H:%M:%S ET")
    return output[-2000:]

def manual_force_close():
    """Manually force-close all positions."""
    ok, output = run_script("alpha_intraday.py", ["--force-close"])
    return output[-2000:]

def manual_run_sentiment():
    """Manually trigger sentiment fetcher."""
    ok, output = run_script("realtime_news.py")
    return output[-2000:]


# ─── Gradio Dashboard ────────────────────────────────────────────────────
def build_dashboard():
    """Build the Gradio dashboard UI for intraday trading."""

    with gr.Blocks(
        title="AlphaShariaBot — Intraday Dashboard",
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="blue",
        ),
    ) as app:
        gr.Markdown("# ⚡ AlphaShariaBot — Intraday Day Trading Dashboard")
        gr.Markdown("Halal intraday trading with AI signals, VWAP entries, "
                    "and dynamic risk management.")

        with gr.Tabs():
            # Tab 1: Portfolio Status
            with gr.Tab("📊 Portfolio"):
                wallet_display = gr.Textbox(
                    label="Wallet Status",
                    value=load_wallet,
                    lines=30,
                    interactive=False,
                    every=15,  # refresh every 15 sec (more frequent for intraday)
                )
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                refresh_btn.click(load_wallet, outputs=wallet_display)

            # Tab 2: Scheduler & Controls
            with gr.Tab("⏰ Scheduler"):
                sched_display = gr.Textbox(
                    label="Scheduler Status",
                    value=get_scheduler_status,
                    lines=10,
                    interactive=False,
                    every=15,
                )
                with gr.Row():
                    scan_btn = gr.Button("🔍 Run Scan Now", variant="primary")
                    close_btn = gr.Button("🔔 Force-Close All",
                                         variant="stop")
                    sent_btn = gr.Button("📰 Run Sentiment", variant="secondary")
                output_display = gr.Textbox(
                    label="Last Script Output",
                    lines=15,
                    interactive=False,
                )
                scan_btn.click(manual_run_scan, outputs=output_display)
                close_btn.click(manual_force_close, outputs=output_display)
                sent_btn.click(manual_run_sentiment, outputs=output_display)

            # Tab 3: Risk Config
            with gr.Tab("🛡️ Risk"):
                risk_display = gr.Textbox(
                    label="Risk Configuration",
                    value=load_risk_config,
                    lines=12,
                    interactive=False,
                )

            # Tab 4: Logs
            with gr.Tab("📜 Logs"):
                log_display = gr.Textbox(
                    label="Recent Trading Logs",
                    value=load_logs,
                    lines=30,
                    interactive=False,
                    every=15,
                )

    return app


# ─── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start the background scheduler in a daemon thread
    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()
    slog.info("⚡ Starting Intraday Gradio dashboard...")

    # Launch the Gradio app
    app = build_dashboard()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
