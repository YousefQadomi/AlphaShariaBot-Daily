"""
app.py — AlphaShariaBot Cloud Scheduler + Dashboard (Day Trading V2)
=====================================================================
Runs on Hugging Face Spaces (Gradio).
- Intraday trading: Adaptive scan intervals (60s opening, 5min normal, 10min midday)
- News polling:     Every 15 min during market hours
- Force-close:      At 3:50 PM ET
- Dashboard:        Professional dark-themed UI with rich HTML cards
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import shutil
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
BACKUP_DIR  = os.path.join(BASE_DIR, "data", "live", "backups")

# ─── Logging ──────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCHEDULER] %(message)s",
    handlers=[
        logging.FileHandler(SCHED_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
slog = logging.getLogger("scheduler")

# ─── Data Persistence: Backup wallet on startup ──────────────────────────
def backup_wallet():
    """Create a timestamped backup of the wallet file on startup."""
    if os.path.exists(WALLET_PATH):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"wallet_{ts}.json")
        shutil.copy2(WALLET_PATH, backup_path)
        slog.info(f"💾 Wallet backed up → {backup_path}")
        # Keep only last 20 backups
        backups = sorted([
            f for f in os.listdir(BACKUP_DIR) if f.startswith("wallet_")
        ])
        for old in backups[:-20]:
            os.remove(os.path.join(BACKUP_DIR, old))

backup_wallet()

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

# ─── Load wallet JSON safely ─────────────────────────────────────────────
def _load_wallet_json():
    """Load raw wallet dict from disk."""
    try:
        with open(WALLET_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ─── CSS Styles ───────────────────────────────────────────────────────────
CSS = """
<style>
.card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px;
  padding: 20px; margin: 8px 0; border: 1px solid #2a3a5e; }
.card h3 { color: #00d4aa; margin: 0 0 12px 0; font-size: 16px; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.stat-box { background: rgba(0,212,170,0.08); border-radius: 8px; padding: 14px;
  border: 1px solid rgba(0,212,170,0.15); text-align: center; }
.stat-box .label { color: #8899aa; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.stat-box .value { color: #e0e6ed; font-size: 22px; font-weight: 700; margin-top: 4px; }
.stat-box .value.green { color: #00d4aa; }
.stat-box .value.red { color: #ff4757; }
.pos-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.pos-table th { background: rgba(0,212,170,0.12); color: #00d4aa; padding: 8px 12px;
  text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.pos-table td { padding: 8px 12px; border-bottom: 1px solid #2a3a5e; color: #c0c8d4; font-size: 13px; }
.pos-table tr:hover { background: rgba(0,212,170,0.04); }
.pnl-pos { color: #00d4aa; font-weight: 600; }
.pnl-neg { color: #ff4757; font-weight: 600; }
.engine-on { background: linear-gradient(135deg, #00d4aa22, #00d4aa08); border: 1px solid #00d4aa44;
  border-radius: 8px; padding: 12px; text-align: center; color: #00d4aa; font-weight: 600; }
.engine-off { background: linear-gradient(135deg, #ff475722, #ff475708); border: 1px solid #ff475744;
  border-radius: 8px; padding: 12px; text-align: center; color: #ff4757; font-weight: 600; }
.summary-bar { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }
.summary-item { background: rgba(255,255,255,0.04); border-radius: 6px; padding: 8px 16px; }
.summary-item .s-label { color: #667; font-size: 11px; }
.summary-item .s-val { color: #e0e6ed; font-size: 15px; font-weight: 600; }
</style>
"""

# ─── Portfolio HTML ──────────────────────────────────────────────────────
def load_portfolio_html():
    """Render portfolio as rich HTML cards."""
    w = _load_wallet_json()
    if w is None:
        return CSS + '<div class="card"><h3>No Data</h3><p style="color:#8899aa">No wallet file found. Start the engine to begin trading.</p></div>'

    cash = w.get('cash', 0)
    pnl = w.get('realized_pnl', 0)
    initial = w.get('initial_balance', 1000)
    equity = initial + pnl
    positions = w.get('positions', [])
    history = w.get('trade_history', [])
    pnl_class = 'green' if pnl >= 0 else 'red'
    pnl_sign = '+' if pnl >= 0 else ''
    pnl_pct = (pnl / initial * 100) if initial > 0 else 0

    # Stats cards
    html = CSS + f'''
    <div class="card">
      <h3>💰 Account Overview</h3>
      <div class="stat-grid">
        <div class="stat-box"><div class="label">Equity</div><div class="value">${equity:,.2f}</div></div>
        <div class="stat-box"><div class="label">Cash Available</div><div class="value">${cash:,.2f}</div></div>
        <div class="stat-box"><div class="label">Realized P&L</div><div class="value {pnl_class}">{pnl_sign}${pnl:,.4f}</div></div>
        <div class="stat-box"><div class="label">Return</div><div class="value {pnl_class}">{pnl_sign}{pnl_pct:.2f}%</div></div>
        <div class="stat-box"><div class="label">Open Positions</div><div class="value">{len(positions)}</div></div>
        <div class="stat-box"><div class="label">Total Trades</div><div class="value">{len(history)}</div></div>
      </div>
    </div>
    '''

    # Open positions table
    if positions:
        html += '<div class="card"><h3>📂 Open Positions</h3>'
        html += '<table class="pos-table"><thead><tr>'
        html += '<th>Ticker</th><th>Shares</th><th>Entry Price</th><th>Cost</th><th>Score</th><th>Entry Time</th>'
        html += '</tr></thead><tbody>'
        for p in positions:
            score = p.get('entry_score', 0)
            html += f'''<tr>
              <td><strong>{p["ticker"]}</strong></td>
              <td>{p["shares"]:.4f}</td>
              <td>${p["entry_price"]:,.2f}</td>
              <td>${p.get("entry_cost", 0):,.2f}</td>
              <td>{score:.0f}</td>
              <td>{p.get("entry_time", "?")}</td>
            </tr>'''
        html += '</tbody></table></div>'
    else:
        html += '<div class="card"><h3>📂 Open Positions</h3><p style="color:#667">No open positions. The bot will buy when it finds high-scoring opportunities.</p></div>'

    # Win rate
    if history:
        wins = sum(1 for t in history if t['pnl'] > 0)
        total = len(history)
        avg_pnl = sum(t['pnl'] for t in history) / total
        win_rate = wins / total * 100
        wr_class = 'green' if win_rate >= 50 else 'red'
        html += f'''<div class="card"><h3>📊 Performance</h3>
          <div class="stat-grid">
            <div class="stat-box"><div class="label">Win Rate</div><div class="value {wr_class}">{win_rate:.0f}%</div></div>
            <div class="stat-box"><div class="label">Wins / Total</div><div class="value">{wins} / {total}</div></div>
            <div class="stat-box"><div class="label">Avg P&L/Trade</div><div class="value {"green" if avg_pnl >= 0 else "red"}">${avg_pnl:+.4f}</div></div>
          </div></div>'''

    return html

# ─── Trade History HTML ──────────────────────────────────────────────────
def load_trade_history_html():
    """Render trade history as a rich HTML table."""
    w = _load_wallet_json()
    if w is None:
        return CSS + '<div class="card"><p style="color:#667">No wallet file found.</p></div>'

    history = w.get('trade_history', [])
    if not history:
        return CSS + '<div class="card"><h3>📜 Trade History</h3><p style="color:#667">No completed trades yet. The bot will log every buy and sell here.</p></div>'

    history = list(reversed(history))  # newest first

    total_pnl = sum(t['pnl'] for t in history)
    wins = sum(1 for t in history if t['pnl'] > 0)
    losses = sum(1 for t in history if t['pnl'] < 0)
    total = len(history)
    win_rate = (wins / total * 100) if total > 0 else 0
    avg_pnl = total_pnl / total if total > 0 else 0

    tp_class = 'green' if total_pnl >= 0 else 'red'
    tp_sign = '+' if total_pnl >= 0 else ''

    html = CSS + f'''
    <div class="card">
      <h3>📊 Trade Summary</h3>
      <div class="stat-grid">
        <div class="stat-box"><div class="label">Total Trades</div><div class="value">{total}</div></div>
        <div class="stat-box"><div class="label">Wins</div><div class="value green">{wins}</div></div>
        <div class="stat-box"><div class="label">Losses</div><div class="value red">{losses}</div></div>
        <div class="stat-box"><div class="label">Win Rate</div><div class="value {"green" if win_rate >= 50 else "red"}">{win_rate:.0f}%</div></div>
        <div class="stat-box"><div class="label">Total P&L</div><div class="value {tp_class}">{tp_sign}${total_pnl:,.4f}</div></div>
        <div class="stat-box"><div class="label">Avg P&L/Trade</div><div class="value {"green" if avg_pnl >= 0 else "red"}">${avg_pnl:+.4f}</div></div>
      </div>
    </div>
    '''

    # Trade table
    html += '<div class="card"><h3>📜 All Trades</h3>'
    html += '<table class="pos-table"><thead><tr>'
    html += '<th>#</th><th>Ticker</th><th>Action</th><th>Entry $</th><th>Exit $</th>'
    html += '<th>Shares</th><th>P&L</th><th>Return</th><th>Time</th>'
    html += '</tr></thead><tbody>'

    for i, t in enumerate(history, 1):
        pnl = t.get('pnl', 0)
        ret = t.get('return_pct', 0)
        pnl_cls = 'pnl-pos' if pnl > 0 else ('pnl-neg' if pnl < 0 else '')
        icon = '✅' if pnl > 0 else ('❌' if pnl < 0 else '➖')
        reason = t.get('exit_reason', '?')
        exit_time = t.get('exit_time', '?')
        # Show just time portion for compactness
        time_short = exit_time.split(' ')[-1] if ' ' in str(exit_time) else exit_time
        date_short = exit_time.split(' ')[0] if ' ' in str(exit_time) else ''

        html += f'''<tr>
          <td>{i}</td>
          <td><strong>{t.get("ticker", "?")}</strong></td>
          <td>{icon} {reason}</td>
          <td>${t.get("entry_price", 0):,.2f}</td>
          <td>${t.get("exit_price", 0):,.2f}</td>
          <td>{t.get("shares", 0):.4f}</td>
          <td class="{pnl_cls}">${pnl:+.4f}</td>
          <td class="{pnl_cls}">{ret:+.2f}%</td>
          <td>{date_short}<br><small>{time_short}</small></td>
        </tr>'''

    html += '</tbody></table></div>'

    # Best/worst
    if history:
        best = max(history, key=lambda t: t.get('pnl', 0))
        worst = min(history, key=lambda t: t.get('pnl', 0))
        html += f'''<div class="card">
          <h3>🏆 Notable Trades</h3>
          <div class="stat-grid">
            <div class="stat-box"><div class="label">Best Trade</div>
              <div class="value green">{best["ticker"]} +${best["pnl"]:,.4f} ({best.get("return_pct",0):+.2f}%)</div></div>
            <div class="stat-box"><div class="label">Worst Trade</div>
              <div class="value red">{worst["ticker"]} ${worst["pnl"]:+,.4f} ({worst.get("return_pct",0):+.2f}%)</div></div>
          </div></div>'''

    return html

# ─── Scheduler Status HTML ───────────────────────────────────────────────
def get_scheduler_html():
    s = scheduler_status
    et_now = datetime.now(ZoneInfo("America/New_York"))
    is_market = (
        et_now.weekday() < 5 and
        (et_now.hour > 9 or (et_now.hour == 9 and et_now.minute >= 30)) and
        et_now.hour < 16
    )
    market_badge = '<span style="color:#00d4aa">● OPEN</span>' if is_market else '<span style="color:#ff4757">● CLOSED</span>'
    engine_div = f'<div class="engine-on">🟢 ENGINE RUNNING — Scanning automatically</div>' if s['running'] else f'<div class="engine-off">🔴 ENGINE STOPPED — Click Start Engine to begin</div>'

    return CSS + f'''
    {engine_div}
    <div class="card" style="margin-top:12px">
      <h3>⏰ Scheduler Status</h3>
      <div class="stat-grid">
        <div class="stat-box"><div class="label">Market</div><div class="value">{market_badge}</div></div>
        <div class="stat-box"><div class="label">Time (ET)</div><div class="value" style="font-size:16px">{et_now.strftime("%H:%M:%S")}</div></div>
        <div class="stat-box"><div class="label">Last Scan</div><div class="value" style="font-size:14px">{s["scan_last_run"]}</div></div>
        <div class="stat-box"><div class="label">Next Scan</div><div class="value" style="font-size:14px">{s["scan_next_run"]}</div></div>
        <div class="stat-box"><div class="label">Scans Today</div><div class="value">{s["scans_today"]}</div></div>
        <div class="stat-box"><div class="label">News Poll</div><div class="value" style="font-size:14px">{s["news_last_run"]}</div></div>
      </div>
    </div>
    '''

# ─── Risk Config HTML ────────────────────────────────────────────────────
def load_risk_html():
    try:
        with open(RISK_CFG, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return CSS + '<div class="card"><p style="color:#667">No risk config found.</p></div>'

    return CSS + f'''
    <div class="card">
      <h3>🛡️ Risk Parameters</h3>
      <div class="stat-grid">
        <div class="stat-box"><div class="label">Stop Loss (ATR×)</div><div class="value">{cfg.get("stop_loss_atr_multiplier", 1.5)}×</div></div>
        <div class="stat-box"><div class="label">Fallback Stop</div><div class="value red">{cfg.get("stop_loss_fallback_pct", -0.012)*100:.1f}%</div></div>
        <div class="stat-box"><div class="label">Trailing Stop</div><div class="value">{cfg.get("trailing_stop_pct", -0.005)*100:.1f}%</div></div>
        <div class="stat-box"><div class="label">Partial Exit</div><div class="value green">+{cfg.get("partial_exit_pct", 0.008)*100:.1f}%</div></div>
        <div class="stat-box"><div class="label">Full Take Profit</div><div class="value green">+{cfg.get("full_take_profit_pct", 0.02)*100:.1f}%</div></div>
        <div class="stat-box"><div class="label">Time Stop</div><div class="value">{cfg.get("time_stop_minutes", 90)} min</div></div>
        <div class="stat-box"><div class="label">Daily Loss Cap</div><div class="value red">{cfg.get("max_daily_loss_pct", -0.03)*100:.1f}%</div></div>
        <div class="stat-box"><div class="label">Max Trades/Day</div><div class="value">{cfg.get("max_daily_trades", 30)}</div></div>
        <div class="stat-box"><div class="label">Max Consec Losses</div><div class="value">{cfg.get("max_consecutive_losses", 5)}</div></div>
        <div class="stat-box"><div class="label">Risk Per Trade</div><div class="value">{cfg.get("risk_per_trade_pct", 0.01)*100:.1f}%</div></div>
        <div class="stat-box"><div class="label">Max Position %</div><div class="value">{cfg.get("max_position_pct", 0.15)*100:.0f}%</div></div>
        <div class="stat-box"><div class="label">Circuit Breaker</div><div class="value red">{cfg.get("max_drawdown_pct", -0.05)*100:.1f}% DD</div></div>
      </div>
    </div>
    '''

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
    "scan_next_run": "Waiting...",
    "news_last_run": "Never",
    "scans_today": 0,
    "trades_today": 0,
    "last_output": "",
    "running": False,
}
_scheduler_thread = None

# ─── Adaptive Scan Interval ──────────────────────────────────────────────
def _get_adaptive_interval():
    """Return scan interval based on market session phase."""
    now = datetime.now(ZoneInfo("America/New_York"))
    market_open = now.replace(hour=9, minute=30, second=0)
    mins = max(0, (now - market_open).total_seconds() / 60)
    if mins < 15:
        return 30, "first_15min"   # first 15 min: every 30s
    elif mins < 30:
        return 60, "opening"       # rest of opening: every 60s
    elif mins >= 330:
        return 120, "power_hour"   # 3:00 PM+: every 2 min
    elif 120 < mins < 270:
        return 600, "midday"       # 11:30-2:00: every 10 min
    else:
        return 300, "normal"       # default: every 5 min

# ─── Background Scheduler ────────────────────────────────────────────────
def scheduler_loop():
    last_scan_time = 0
    last_news_time = 0
    today_date = None

    slog.info("⚡ Intraday scheduler V2 started (adaptive intervals).")

    while scheduler_status["running"]:
        try:
            now = datetime.now(ZoneInfo("America/New_York"))
            now_ts = time.time()

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

                if now_ts - last_scan_time >= scan_interval:
                    ok, output = run_script("alpha_intraday.py")
                    last_scan_time = now_ts
                    scheduler_status["scan_last_run"] = now.strftime("%H:%M:%S ET")
                    next_scan = now + timedelta(seconds=scan_interval)
                    scheduler_status["scan_next_run"] = f"{next_scan.strftime('%H:%M:%S ET')} ({phase})"
                    scheduler_status["scans_today"] += 1
                    scheduler_status["last_output"] = f"[Scan] {output[-500:]}"

                if now_ts - last_news_time >= 900:
                    last_news_time = now_ts
                    scheduler_status["news_last_run"] = now.strftime("%H:%M:%S ET")

                if now.hour == 15 and now.minute >= 50:
                    slog.info("🔔 EOD Force-close triggered")
                    run_script("alpha_intraday.py", ["--force-close"])

            if now.weekday() == 0 and now.hour == 7 and now.minute < 2:
                slog.info("☪️ Weekly halal universe refresh...")
                run_script("sharia_screener.py")

            sleep_time = 15 if is_market_hours else 60
            time.sleep(sleep_time)

        except Exception as e:
            slog.error(f"Scheduler error: {e}")
            time.sleep(60)

    slog.info("⛔ Scheduler loop stopped.")

# ─── Engine Start / Stop ─────────────────────────────────────────────────
def toggle_engine():
    global _scheduler_thread
    if scheduler_status["running"]:
        scheduler_status["running"] = False
        slog.info("⛔ Engine STOP requested by user.")
        return "⛔ Engine STOPPED."
    else:
        scheduler_status["running"] = True
        _scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        _scheduler_thread.start()
        slog.info("✅ Engine STARTED by user.")
        return "✅ Engine STARTED! Scanning automatically."

# ─── Manual Trigger Functions ────────────────────────────────────────────
def manual_run_scan():
    ok, output = run_script("alpha_intraday.py")
    et_now = datetime.now(ZoneInfo("America/New_York"))
    scheduler_status["scan_last_run"] = et_now.strftime("%H:%M:%S ET")
    return output[-2000:]

def manual_force_close():
    ok, output = run_script("alpha_intraday.py", ["--force-close"])
    return output[-2000:]

def manual_run_sentiment():
    ok, output = run_script("realtime_news.py")
    return output[-2000:]

# ─── Gradio Dashboard ────────────────────────────────────────────────────
def build_dashboard():
    custom_css = """
    .gradio-container { max-width: 1200px !important; }
    .dark { background: #0a0a1a !important; }
    """

    with gr.Blocks(
        title="AlphaShariaBot — Intraday Trading",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        css=custom_css,
    ) as app:
        gr.Markdown(
            "# ⚡ AlphaShariaBot\n"
            "**Halal AI-Powered Intraday Trading** — VWAP entries · ML scoring · Dynamic risk management"
        )

        with gr.Tabs():
            with gr.Tab("📊 Portfolio"):
                portfolio_html = gr.HTML(
                    value=load_portfolio_html,
                    every=15,
                )
                gr.Button("🔄 Refresh", variant="secondary").click(
                    load_portfolio_html, outputs=portfolio_html
                )

            with gr.Tab("⏰ Engine"):
                engine_html = gr.HTML(
                    value=get_scheduler_html,
                    every=10,
                )
                engine_output = gr.Textbox(
                    label="Status", lines=1, interactive=False,
                    value="Click Start Engine to begin."
                )
                with gr.Row():
                    engine_btn = gr.Button("▶️ Start Engine", variant="primary", size="lg")
                    stop_btn = gr.Button("⛔ Stop Engine", variant="stop", size="lg")
                engine_btn.click(toggle_engine, outputs=engine_output)
                stop_btn.click(toggle_engine, outputs=engine_output)

                gr.Markdown("### Manual Controls")
                with gr.Row():
                    scan_btn = gr.Button("🔍 Run Scan", variant="primary")
                    close_btn = gr.Button("🔔 Force-Close All", variant="stop")
                    sent_btn = gr.Button("📰 Run News", variant="secondary")
                output_display = gr.Textbox(label="Output", lines=12, interactive=False)
                scan_btn.click(manual_run_scan, outputs=output_display)
                close_btn.click(manual_force_close, outputs=output_display)
                sent_btn.click(manual_run_sentiment, outputs=output_display)

            with gr.Tab("📜 Trade History"):
                history_html = gr.HTML(
                    value=load_trade_history_html,
                    every=30,
                )
                gr.Button("🔄 Refresh", variant="secondary").click(
                    load_trade_history_html, outputs=history_html
                )

            with gr.Tab("🛡️ Risk"):
                risk_html = gr.HTML(
                    value=load_risk_html,
                    every=60,
                )

            with gr.Tab("📋 Logs"):
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
    slog.info("⚡ Starting AlphaShariaBot Dashboard...")
    app = build_dashboard()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
