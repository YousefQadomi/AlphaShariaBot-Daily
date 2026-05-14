"""
app.py — AlphaShariaBot Cloud Scheduler + Dashboard
=====================================================
Runs on Hugging Face Spaces (Gradio).
- Sentiment fetcher:  Every 6 hours
- Trading bot:        Once daily at 9:35 AM ET (5 min after market open)
- Dashboard:          Shows live wallet status, recent trades, risk health
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
WALLET_PATH = os.path.join(BASE_DIR, "data", "live", "virtual_wallet.json")
RISK_CFG    = os.path.join(BASE_DIR, "data", "live", "risk_config.json")
LOG_PATH    = os.path.join(BASE_DIR, "logs", "alpha_live.log")
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
def run_script(script_name):
    """Run a Python script and return (success, output)."""
    script_path = os.path.join(BASE_DIR, "scripts", script_name)
    slog.info(f"▶ Starting {script_name}...")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,  # 10 min max
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            slog.info(f"✅ {script_name} completed successfully.")
        else:
            slog.error(f"❌ {script_name} failed (exit {result.returncode})")
        return result.returncode == 0, output[-2000:]  # last 2000 chars
    except subprocess.TimeoutExpired:
        slog.error(f"⏰ {script_name} timed out (10 min).")
        return False, "Script timed out after 10 minutes."
    except Exception as e:
        slog.error(f"💥 {script_name} error: {e}")
        return False, str(e)

# ─── Load wallet data for dashboard ──────────────────────────────────────
def load_wallet():
    """Load and return wallet state as formatted text."""
    try:
        with open(WALLET_PATH, "r") as f:
            w = json.load(f)
    except FileNotFoundError:
        return "No wallet file found. Run the bot first."

    lines = []
    lines.append(f"💰 Initial Balance:  ${w.get('initial_balance', 0):,.2f}")
    lines.append(f"💵 Cash Available:   ${w.get('cash', 0):,.2f}")
    lines.append(f"📊 Realized PnL:     ${w.get('realized_pnl', 0):+,.2f}")
    lines.append(f"📂 Open Positions:   {len(w.get('positions', []))}/50")
    lines.append(f"📜 Total Trades:     {len(w.get('trade_history', []))}")
    lines.append(f"📅 Last Run:         {w.get('last_run_date', 'Never')}")

    # Positions table
    positions = w.get("positions", [])
    if positions:
        lines.append(f"\n{'─'*50}")
        lines.append("OPEN POSITIONS:")
        lines.append(f"{'Ticker':<8} {'Shares':>10} {'Entry $':>10} {'Days':>5}")
        lines.append(f"{'─'*8} {'─'*10} {'─'*10} {'─'*5}")
        for p in positions[:20]:  # show first 20
            lines.append(f"{p['ticker']:<8} {p['shares']:>10.4f} ${p['entry_price']:>9.2f} {p.get('days_held', 0):>5}")
        if len(positions) > 20:
            lines.append(f"  ... and {len(positions) - 20} more positions")

    # Recent trades
    history = w.get("trade_history", [])
    if history:
        lines.append(f"\n{'─'*50}")
        lines.append("RECENT TRADES (last 10):")
        lines.append(f"{'Ticker':<8} {'PnL':>10} {'Return':>8} {'Exit':>12}")
        lines.append(f"{'─'*8} {'─'*10} {'─'*8} {'─'*12}")
        for t in history[-10:]:
            pnl_str = f"${t['pnl']:+.2f}"
            ret_str = f"{t.get('return_pct', 0):+.1f}%"
            lines.append(f"{t['ticker']:<8} {pnl_str:>10} {ret_str:>8} {t.get('exit_date', '?'):>12}")

    return "\n".join(lines)

def load_risk_config():
    """Load risk config as formatted text."""
    try:
        with open(RISK_CFG, "r") as f:
            cfg = json.load(f)
        lines = ["RISK MANAGEMENT CONFIG:"]
        lines.append(f"  Stop-loss:        {cfg.get('stop_loss_pct', -0.10) * 100:.0f}%")
        lines.append(f"  Max correlation:  {cfg.get('max_correlation', 0.7)}")
        lines.append(f"  Max drawdown:     {cfg.get('max_drawdown_pct', -0.25) * 100:.0f}%")
        lines.append(f"  Resume at:        {cfg.get('resume_drawdown_pct', -0.18) * 100:.0f}%")
        return "\n".join(lines)
    except FileNotFoundError:
        return "No risk config found."

def load_logs():
    """Load last 50 lines of the trading log."""
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-50:])
    except FileNotFoundError:
        return "No log file found. Run the bot first."

# ─── Scheduler State ─────────────────────────────────────────────────────
scheduler_status = {
    "sentiment_last_run": "Never",
    "trading_last_run": "Never",
    "sentiment_next_run": "Starting...",
    "trading_next_run": "Starting...",
    "last_output": "",
    "running": True,
}

def get_scheduler_status():
    """Return scheduler status as formatted text."""
    s = scheduler_status
    et_now = datetime.now(ZoneInfo("America/New_York"))
    lines = [
        f"🕐 Current Time (ET):     {et_now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"📰 Sentiment Last Run:    {s['sentiment_last_run']}",
        f"📰 Sentiment Next Run:    {s['sentiment_next_run']}",
        f"📈 Trading Last Run:      {s['trading_last_run']}",
        f"📈 Trading Next Run:      {s['trading_next_run']}",
        f"🔄 Scheduler Active:      {'YES' if s['running'] else 'NO'}",
    ]
    return "\n".join(lines)


# ─── Background Scheduler ────────────────────────────────────────────────
def scheduler_loop():
    """
    Background thread that:
    - Runs sentiment_fetcher every 6 hours
    - Runs alpha_live once daily at 9:35 AM ET on weekdays
    """
    SENTIMENT_INTERVAL = 6 * 3600  # 6 hours in seconds
    last_sentiment_run = 0
    last_trading_date = None

    slog.info("🚀 Scheduler started.")

    while scheduler_status["running"]:
        try:
            now = datetime.now(ZoneInfo("America/New_York"))
            now_ts = time.time()

            # ── Sentiment: every 6 hours ──────────────────────────────
            if now_ts - last_sentiment_run >= SENTIMENT_INTERVAL:
                ok, output = run_script("sentiment_fetcher.py")
                last_sentiment_run = now_ts
                scheduler_status["sentiment_last_run"] = now.strftime("%Y-%m-%d %H:%M ET")
                next_run = now + timedelta(hours=6)
                scheduler_status["sentiment_next_run"] = next_run.strftime("%Y-%m-%d %H:%M ET")
                scheduler_status["last_output"] = f"[Sentiment] {output[-500:]}"

            # ── Trading: once at ~9:35 AM ET, weekdays only ───────────
            today = now.date()
            is_weekday = now.weekday() < 5  # Mon-Fri
            is_after_open = now.hour > 9 or (now.hour == 9 and now.minute >= 35)
            not_yet_run_today = last_trading_date != today

            if is_weekday and is_after_open and not_yet_run_today:
                slog.info(f"📈 Market open detected ({now.strftime('%H:%M ET')}). Running trading bot...")
                ok, output = run_script("alpha_live.py")
                last_trading_date = today
                scheduler_status["trading_last_run"] = now.strftime("%Y-%m-%d %H:%M ET")
                # Calculate next trading day
                next_day = today + timedelta(days=1)
                while next_day.weekday() >= 5:  # skip weekends
                    next_day += timedelta(days=1)
                scheduler_status["trading_next_run"] = f"{next_day} ~09:35 ET"
                scheduler_status["last_output"] = f"[Trading] {output[-500:]}"
            elif not is_after_open and is_weekday:
                scheduler_status["trading_next_run"] = f"{today} ~09:35 ET"

            # Sleep 60 seconds between checks
            time.sleep(60)

        except Exception as e:
            slog.error(f"Scheduler error: {e}")
            time.sleep(60)


# ─── Manual Trigger Functions ────────────────────────────────────────────
def manual_run_sentiment():
    """Manually trigger sentiment fetcher."""
    ok, output = run_script("sentiment_fetcher.py")
    et_now = datetime.now(ZoneInfo("America/New_York"))
    scheduler_status["sentiment_last_run"] = et_now.strftime("%Y-%m-%d %H:%M ET")
    return output[-2000:]

def manual_run_trading():
    """Manually trigger trading bot."""
    ok, output = run_script("alpha_live.py")
    et_now = datetime.now(ZoneInfo("America/New_York"))
    scheduler_status["trading_last_run"] = et_now.strftime("%Y-%m-%d %H:%M ET")
    return output[-2000:]


# ─── Gradio Dashboard ────────────────────────────────────────────────────
def build_dashboard():
    """Build the Gradio dashboard UI."""

    with gr.Blocks(
        title="AlphaShariaBot — Dashboard",
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="blue",
        ),
    ) as app:
        gr.Markdown("# ☪️ AlphaShariaBot — Live Dashboard")
        gr.Markdown("Halal stock trading bot with AI-powered ranking & risk management.")

        with gr.Tabs():
            # Tab 1: Portfolio Status
            with gr.Tab("📊 Portfolio"):
                wallet_display = gr.Textbox(
                    label="Wallet Status",
                    value=load_wallet,
                    lines=25,
                    interactive=False,
                    every=30,  # auto-refresh every 30 sec
                )
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                refresh_btn.click(load_wallet, outputs=wallet_display)

            # Tab 2: Scheduler
            with gr.Tab("⏰ Scheduler"):
                sched_display = gr.Textbox(
                    label="Scheduler Status",
                    value=get_scheduler_status,
                    lines=8,
                    interactive=False,
                    every=30,
                )
                with gr.Row():
                    sent_btn = gr.Button("📰 Run Sentiment Now", variant="primary")
                    trade_btn = gr.Button("📈 Run Trading Now", variant="primary")
                output_display = gr.Textbox(
                    label="Last Script Output",
                    lines=15,
                    interactive=False,
                )
                sent_btn.click(manual_run_sentiment, outputs=output_display)
                trade_btn.click(manual_run_trading, outputs=output_display)

            # Tab 3: Risk Config
            with gr.Tab("🛡️ Risk"):
                risk_display = gr.Textbox(
                    label="Risk Configuration",
                    value=load_risk_config,
                    lines=8,
                    interactive=False,
                )

            # Tab 4: Logs
            with gr.Tab("📜 Logs"):
                log_display = gr.Textbox(
                    label="Recent Trading Logs",
                    value=load_logs,
                    lines=30,
                    interactive=False,
                    every=30,
                )

    return app


# ─── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start the background scheduler in a daemon thread
    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()
    slog.info("📊 Starting Gradio dashboard...")

    # Launch the Gradio app
    app = build_dashboard()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
