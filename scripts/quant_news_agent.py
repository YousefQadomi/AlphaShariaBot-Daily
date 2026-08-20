import os
import sys
import json
import time
import asyncio
import aiohttp
from aiohttp import web
import feedparser
import yfinance as yf
import pandas as pd
import re
from datetime import datetime
import pytz
from dotenv import load_dotenv, set_key
from google import genai
from google.genai import types
import chromadb
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-3.6-flash'

OUTPUT_FILE = os.path.join(BASE_DIR, "ai_news_terminal", "public", "omni_data.json")
HALAL_STOCKS_FILE = os.path.join(BASE_DIR, "data", "halal_stocks.csv")
SWARM_SIGNALS_FILE = os.path.join(BASE_DIR, "data", "live", "gemini_reports.json")

# Initialize AI Tools
sia = SentimentIntensityAnalyzer()
chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "data", "chroma_db"))
collection = chroma_client.get_or_create_collection(name="news_memory")

seen_guids = set()
halal_tickers = set()
all_historic_news = []
swarm_signals = []
bot_state = "RUNNING" # "RUNNING", "PAUSED", "ASLEEP"
macro_halt = False
paper_trades = []
agent_heartbeats = {"macro": 0, "insider": 0, "execution": 0, "master": 0, "filter": 0, "domino": 0, "scouts": 0}
pending_execution_queue = []

# Queues for Swarm Architecture
raw_news_queue = asyncio.Queue()
filter_queue = asyncio.Queue()
master_queue = asyncio.Queue()
execution_queue = asyncio.Queue()

# Scout Configuration
RSS_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "Investing": "https://www.investing.com/rss/news_25.rss",
    "Benzinga": "https://www.benzinga.com/feed",
    "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
}
REDDIT_SUBS = ["wallstreetbets", "stocks", "investing", "StockMarket", "pennystocks"]

def load_halal_stocks():
    global halal_tickers
    try:
        if os.path.exists(HALAL_STOCKS_FILE):
            df = pd.read_csv(HALAL_STOCKS_FILE)
            df.columns = df.columns.str.lower()
            col = 'ticker' if 'ticker' in df.columns else 'symbol'
            halal_tickers = set(df[col].str.upper().tolist())
            print(f"🕌 [SHARIA] Loaded {len(halal_tickers)} Halal Tickers.")
    except Exception as e:
        print(f"⚠️ Failed to load halal stocks: {e}")

# ==========================================
# 🌐 WEB API & TELEGRAM CONTROL
# ==========================================

async def change_bot_state(new_state):
    global bot_state
    if new_state in ["RUNNING", "PAUSED", "ASLEEP"]:
        bot_state = new_state
        print(f"🎮 [CONTROL] State changed to {bot_state}")
        if bot_state != "RUNNING":
            while not raw_news_queue.empty():
                raw_news_queue.get_nowait()
                raw_news_queue.task_done()
            while not filter_queue.empty():
                filter_queue.get_nowait()
                filter_queue.task_done()
            while not master_queue.empty():
                master_queue.get_nowait()
                master_queue.task_done()
                
        payload = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "news": all_historic_news,
            "telegram_status": "connected" if TELEGRAM_CHAT_ID else "waiting",
            "bot_state": bot_state,
            "macro_halt": macro_halt,
            "paper_trades": paper_trades
        }
        await asyncio.to_thread(_save_json, payload)

async def handle_status_get(request):
    return web.json_response({"state": bot_state})

async def handle_status_post(request):
    global bot_state
    try:
        data = await request.json()
        new_state = data.get("state")
        await change_bot_state(new_state)
    except:
        pass
    return web.json_response({"state": bot_state})

async def start_web_server():
    app = web.Application()
    
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == 'OPTIONS':
            response = web.Response()
        else:
            response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    app.middlewares.append(cors_middleware)
    app.router.add_get('/api/status', handle_status_get)
    app.router.add_post('/api/status', handle_status_post)
    app.router.add_options('/api/status', handle_status_get)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    print("🌐 [API] Web Control Server listening on port 8080")

async def send_telegram_alert(session, text):
    if not TELEGRAM_CHAT_ID or not TELEGRAM_BOT_TOKEN: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=5)
    except:
        pass

async def auto_fetch_telegram_chat_id(session):
    global TELEGRAM_CHAT_ID
    if TELEGRAM_CHAT_ID: return True
    if not TELEGRAM_BOT_TOKEN: return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        async with session.get(url, timeout=5) as res:
            data = await res.json()
            if data.get("ok") and len(data["result"]) > 0:
                chat_id = str(data["result"][0]["message"]["chat"]["id"])
                TELEGRAM_CHAT_ID = chat_id
                set_key(ENV_PATH, "TELEGRAM_CHAT_ID", chat_id)
                await send_telegram_alert(session, "✅ تم تفعيل العصبون المركزي (Neural Link) لـ رادار ألفا!")
                return True
    except:
        pass
    return False

async def telegram_listener(session):
    global bot_state
    last_update_id = 0
    while True:
        if TELEGRAM_BOT_TOKEN:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={last_update_id}&timeout=10"
                async with session.get(url, timeout=15) as res:
                    data = await res.json()
                    if data.get("ok"):
                        for update in data["result"]:
                            last_update_id = update["update_id"] + 1
                            if "message" in update and "text" in update["message"]:
                                text = update["message"]["text"].lower()
                                if text in ["/sleep", "/pause"]:
                                    await change_bot_state("PAUSED")
                                    await send_telegram_alert(session, "🛑 تم استلام الأمر: المنظومة الآن في وضع السبات (إيقاف) وتفريغ جميع المهام.")
                                elif text in ["/wake", "/start", "/resume"]:
                                    await change_bot_state("RUNNING")
                                    await send_telegram_alert(session, "🟢 تم استلام الأمر: استيقاظ المنظومة، جاري مسح الأسواق!")
                                elif text == "/status":
                                    await send_telegram_alert(session, f"ℹ️ حالة المنظومة الحالية: {bot_state}")
            except:
                pass
        await asyncio.sleep(5)

async def smart_schedule():
    global bot_state
    ny_tz = pytz.timezone('America/New_York')
    while True:
        if bot_state != "PAUSED": # Only automate if not manually paused by user
            now_ny = datetime.now(ny_tz)
            is_weekend = now_ny.weekday() >= 5
            is_market_hours = 8 <= now_ny.hour < 17
            
            if is_weekend or not is_market_hours:
                if bot_state == "RUNNING":
                    print("🌙 [SCHEDULE] Market closed. Going to sleep.")
                    await change_bot_state("ASLEEP")
            else:
                if bot_state == "ASLEEP":
                    print("☀️ [SCHEDULE] Market opening soon. Waking up!")
                    await change_bot_state("RUNNING")
        await asyncio.sleep(60)

# ==========================================
# 🕵️‍♂️ SCOUT AGENTS
# ==========================================

async def rss_scout(session, source_name, url):
    print(f"🕵️‍♂️ [SCOUT] {source_name} agent deployed.")
    while True:
        if bot_state != "RUNNING":
            await asyncio.sleep(10)
            continue
        try:
            async with session.get(url, timeout=10) as response:
                content = await response.text()
                feed = await asyncio.to_thread(feedparser.parse, content)
                for entry in feed.entries[:3]:
                    guid = entry.get("id", entry.link)
                    if guid not in seen_guids:
                        seen_guids.add(guid)
                        item = {"source": source_name, "title": entry.title, "summary": entry.get("summary", "")[:250], "guid": guid}
                        await raw_news_queue.put(item)
        except:
            pass
        await asyncio.sleep(60)

async def reddit_scout(session, sub):
    print(f"🕵️‍♂️ [SCOUT] r/{sub} agent deployed.")
    headers = {'User-agent': f'AlphaSwarm_Agent_{sub}_v6.0'}
    url = f"https://www.reddit.com/r/{sub}/new.json?limit=3"
    while True:
        if bot_state != "RUNNING":
            await asyncio.sleep(10)
            continue
        try:
            async with session.get(url, headers=headers, timeout=10) as res:
                if res.status == 200:
                    data = await res.json()
                    for post in data['data']['children']:
                        post_data = post['data']
                        guid = post_data['name']
                        if guid not in seen_guids:
                            seen_guids.add(guid)
                            item = {"source": f"Reddit r/{sub}", "title": post_data['title'], "summary": post_data.get('selftext', '')[:250], "guid": guid}
                            await raw_news_queue.put(item)
        except:
            pass
        await asyncio.sleep(60)

# ==========================================
# 🔮 DOMINO PREDICTOR AGENT & LOCAL FILTER
# ==========================================

FINANCIAL_KEYWORDS = {'EARNINGS', 'INFLATION', 'RATES', 'FED', 'BULL', 'BEAR', 'ACQUISITION', 'MERGER', 'BANKRUPTCY', 'CEO', 'REVENUE', 'PROFIT', 'DIVIDEND', 'Q1', 'Q2', 'Q3', 'Q4', 'STOCKS', 'SHARES', 'MARKET', 'MARKETS', 'PRICE', 'SALES', 'GROWTH', 'COMPANY', 'INVEST', 'INVESTING', 'INVESTORS', 'TRADING', 'TRADE', 'WALL STREET', 'FINANCE', 'ECONOMY', 'TECH', 'AI', 'INDEX', 'S&P', 'NASDAQ', 'DOW', 'FUNDS'}

def local_fast_filter(text):
    text_upper = text.upper()
    words = set(re.findall(r'\b[A-Z]{2,5}\b', text_upper))
    if words.intersection(halal_tickers):
        return True
    for w in FINANCIAL_KEYWORDS:
        if w in text_upper:
            return True
    return False

def _gemini_extract_domino_and_direct(text):
    prompt = f"Extract official US stock market ticker symbols from this text. Return ONLY a JSON object with two arrays: 'direct' (tickers directly mentioned) and 'domino' (up to 3 tickers heavily affected by systemic risk/domino effect). Return ONLY valid JSON: {{\"direct\": [], \"domino\": []}}. Text: {text}"
    try:
        res = client.models.generate_content(model=MODEL_ID, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
        return json.loads(res.text)
    except:
        return {'direct': [], 'domino': []}

async def domino_agent():
    print("🔮 [DOMINO] Systemic Risk Prediction Agent deployed.")
    while True:
        agent_heartbeats["domino"] = time.time()
        try:
            item = await asyncio.wait_for(raw_news_queue.get(), timeout=2.0)
            
            summary_text = item['title'] + " " + item['summary']
            vader_score = sia.polarity_scores(summary_text)['compound']
            
            past_context = ""
            if not local_fast_filter(summary_text):
                raw_news_queue.task_done()
                continue
                
            extraction = await asyncio.to_thread(_gemini_extract_domino_and_direct, summary_text)
            
            item['tickers'] = extraction.get('direct', [])
            item['predicted_domino'] = extraction.get('domino', [])
            
            await filter_queue.put(item)
            raw_news_queue.task_done()
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(2)

# ==========================================
# 🧹 SHARIA FILTER AGENT
# ==========================================

async def filter_agent():
    print("🧹 [CLEANER] Sharia Filter Agent deployed.")
    while True:
        item = await filter_queue.get()
        
        all_tickers = item['tickers'] + item['predicted_domino']
        halal_involved = False
        tagged_tickers = []
        
        for t in item['tickers']:
            if t.upper() in halal_tickers:
                tagged_tickers.append(f"{t} (✅ إسلامي)")
                halal_involved = True
            else:
                tagged_tickers.append(f"{t} (❌ محرم)")
                
        for t in item['predicted_domino']:
            if t.upper() in halal_tickers:
                halal_involved = True
        
        if not (all_tickers and not halal_involved):
            item['tagged_tickers'] = tagged_tickers
            await master_queue.put(item)
            
        filter_queue.task_done()

# ==========================================
# 🧠 MASTER QUANT AGENT
# ==========================================

def _fetch_quant_data(tickers):
    data = {}
    for ticker in tickers[:3]:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            hist = tk.history(period="1mo")
            pcr = "N/A"
            try:
                opts = tk.options
                if opts:
                    chain = tk.option_chain(opts[0])
                    calls_vol = chain.calls['volume'].sum()
                    puts_vol = chain.puts['volume'].sum()
                    if calls_vol > 0: pcr = round(puts_vol / calls_vol, 2)
            except:
                pass

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                data[ticker] = {
                    "price": round(current_price, 2),
                    "PCR": pcr,
                    "PE": info.get("trailingPE", "N/A")
                }
        except:
            pass
    return data

def _gemini_deep_analysis(news_item, quant_data, vader_score, past_context):
    prompt = f"""
    أنت العقل المدبر لـ "منظومة السرب الذكي". 
    هذا الخبر المفلتر: {news_item['title']} - {news_item['summary']}
    
    الأسهم المتأثرة المتوقعة (الدومينو) من وكيل الدومينو: {json.dumps(news_item['predicted_domino'])}
    الأرقام الحية وسوق الخيارات (PCR>1 سلبي، <1 إيجابي): {json.dumps(quant_data)}
    التقييم اللغوي الرياضي للخبر (VADER): {vader_score} (من -1 لسلبي جداً، لـ 1 لإيجابي جداً)
    {past_context}
    
    قواعد بوت التداول (Alpha Intraday) - أجب بصرامة علمية:
    - VETO: إذا كان (VADER سلبي جداً) و(PCR عالي) ويهدد السهم أو يؤثر كدومينو سلبي.
    - BOOST: إذا كان (VADER إيجابي جداً) ومحفز للنمو.
    - NEUTRAL: غير حاسم أو إذا كان تأثير الخبر لحظي ومؤقت (Whipsaw).

    قاعدة هامة جداً (المدة الزمنية):
    إذا كان الخبر ضجيج أو تأثيره لحظي مؤقت (مثل تغريدة عابرة، خطاب مفاجئ)، فيجب أن يكون impact_duration هو "لحظي مؤقت (Transient)" و bot_action يجب أن يكون إجبارياً "NEUTRAL" لمنع بوت التداول من التخبط، وقم بوضع volatility_warning = true. فقط الأخبار التي لها تأثير هيكلي أو طويل الأمد يسمح لها بأخذ VETO أو BOOST.

    أرجع JSON حصراً:
    - "source": {news_item['source']}
    - "title_ar": عنوان عربي جذاب
    - "event_summary": ملخص
    - "predictive_study": دراسة استشرافية تربط الأرقام (الخيارات والمشاعر) بالحدث وتأثير الدومينو.
    - "domino_tickers": مصفوفة (Array) تحتوي على رموز أسهم الدومينو المؤكدة والمنقحة.
    - "domino_explanation": شرح تأثير الدومينو.
    - "impact_score": من -100 إلى 100.
    - "urgency": "عاجل جداً" أو "هام" أو "مراقبة".
    - "impact_duration": حدد واحدة (لحظي مؤقت (Transient) | قصير الأمد (1-3 أيام) | طويل الأمد).
    - "trade_action_window": حدد متى يتم التداول (مثال: شراء فوري، انتظار الهبوط، بيع فوراً، لا تداول).
    - "volatility_warning": true أو false.
    - "bot_action": إما "VETO" أو "BOOST" أو "NEUTRAL".
    """
    try:
        res = client.models.generate_content(model=MODEL_ID, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
        return json.loads(res.text)
    except:
        return None

async def master_agent(session):
    global all_historic_news
    global swarm_signals
    print("🧠 [MASTER] Neural Link Quant Brain deployed...")
    
    while True:
        if not TELEGRAM_CHAT_ID:
            await auto_fetch_telegram_chat_id(session)
            
        try:
            agent_heartbeats["master"] = time.time()
            item = await asyncio.wait_for(master_queue.get(), timeout=10.0)
            
            summary_text = item['title'] + " " + item['summary']
            vader_score = sia.polarity_scores(summary_text)['compound']
            
            past_context = ""
            try:
                results = collection.query(query_texts=[summary_text], n_results=2)
                if results and results.get('documents') and len(results['documents']) > 0 and results['documents'][0]:
                    past_context = "السياق التاريخي القريب: " + " | ".join(results['documents'][0])
            except:
                pass
                
            try:
                collection.add(documents=[summary_text], ids=[item['guid']])
            except:
                pass
            
            q_data = await asyncio.to_thread(_fetch_quant_data, item['tickers']) if item['tickers'] else {}
            
            final = await asyncio.to_thread(_gemini_deep_analysis, item, q_data, vader_score, past_context)
            
            if final:
                domino_parts = []
                for dt in final.get("domino_tickers", []):
                    if dt.upper() in halal_tickers:
                        domino_parts.append(f"{dt} (✅ إسلامي)")
                    else:
                        domino_parts.append(f"{dt} (❌ محرم)")
                
                dom_full = final.get("domino_explanation", "")
                if domino_parts:
                    dom_full += f"\n*الأسهم المتأثرة:* {', '.join(domino_parts)}"
                
                final["domino_effect"] = dom_full
                final["analysis_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                final["related_tickers_tagged"] = item['tagged_tickers']
                
                if final["urgency"] == "عاجل جداً" or abs(final["impact_score"]) >= 70:
                    bot_icon = "🛑" if "VETO" in final.get('bot_action', '') else ("🟢" if "BOOST" in final.get('bot_action', '') else "🟡")
                    msg = f"🦅 *مرصد ألفا | أخبار*\n"
                    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
                    msg += f"🚨 *الطوارئ:* {final['urgency']}\n"
                    msg += f"📰 *الحدث:* {final['title_ar']}\n"
                    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
                    msg += f"🎯 *الأسهم:* {', '.join(item['tagged_tickers']) if item['tagged_tickers'] else 'أسواق عامة'}\n"
                    msg += f"📊 *أثر السوق:* {final['impact_score']} / 100\n"
                    msg += f"🤖 *قرار البوت:* {bot_icon} {final.get('bot_action', 'NEUTRAL')}\n"
                    msg += f"⏳ *عمر التأثير:* {final.get('impact_duration', 'غير محدد')}"
                    if final.get('volatility_warning'):
                        msg += " ⚠️ (تقلب لحظي عالي!)"
                    msg += f"\n🎯 *توقيت التنفيذ:* {final.get('trade_action_window', 'لا يوجد')}\n"
                    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
                    msg += f"💡 *التحليل:*\n{final['predictive_study']}\n\n"
                    msg += f"🔗 *الدومينو:*\n{dom_full}"
                    await send_telegram_alert(session, msg)
                    
                    all_involved_tickers = list(set(item['tickers'] + final.get("domino_tickers", [])))
                    
                    if not final.get('volatility_warning'):
                        if "VETO" in final.get('bot_action', ''):
                            for t in item['tickers']:
                                if t.upper() in halal_tickers:
                                    await execution_queue.put({"action": "بيع/تخفيف (VETO)", "ticker": t.upper(), "price": q_data.get(t.upper(), {}).get("price", "Market")})
                        elif "BOOST" in final.get('bot_action', ''):
                            for t in item['tickers']:
                                if t.upper() in halal_tickers:
                                    await execution_queue.put({"action": "شراء (BOOST)", "ticker": t.upper(), "price": q_data.get(t.upper(), {}).get("price", "Market")})

                    if all_involved_tickers:
                        swarm_signals.append({
                            "symbols": all_involved_tickers,
                            "sentiment_score": vader_score if "VETO" not in final.get('bot_action', '') else -1.0
                        })
                        swarm_signals = swarm_signals[-50:]
                        os.makedirs(os.path.dirname(SWARM_SIGNALS_FILE), exist_ok=True)
                        with open(SWARM_SIGNALS_FILE, "w", encoding="utf-8") as sf:
                            json.dump(swarm_signals, sf)
                    
                all_historic_news = [final] + all_historic_news
                all_historic_news = all_historic_news[:30]
                
            master_queue.task_done()
            
        except asyncio.TimeoutError:
            pass
            
        payload = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "news": all_historic_news,
            "telegram_status": "connected" if TELEGRAM_CHAT_ID else "waiting",
            "bot_state": bot_state,
            "macro_halt": macro_halt,
            "paper_trades": paper_trades
        }
        await asyncio.to_thread(_save_json, payload)


async def macro_agent(session):
    global macro_halt, agent_heartbeats
    print("📅 [MACRO] Economic Calendar Sentinel deployed.")
    
    state_file = "data/macro_state.json"
    import os
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                saved = json.load(f)
                macro_halt = saved.get("macro_halt", False)
                if macro_halt:
                    print("🛑 [MACRO HALT] Resumed from saved state (Halted).")
        except:
            pass
            
    last_successful_macro_fetch = time.time()
    
    while True:
        if bot_state != "RUNNING":
            await asyncio.sleep(10)
            continue
        agent_heartbeats["macro"] = time.time()
        
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            async with session.get(url, timeout=10) as res:
                if res.status == 200:
                    last_successful_macro_fetch = time.time()
                    events = await res.json()
                    now = datetime.utcnow()
                    halt_active = False
                    for ev in events:
                        if ev.get("country") == "USD" and ev.get("impact") == "High":
                            ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(pytz.utc).replace(tzinfo=None)
                            time_diff = (ev_time - now).total_seconds()
                            if -900 <= time_diff <= 900:
                                halt_active = True
                                break
                    
                    if halt_active and not macro_halt:
                        macro_halt = True
                        with open(state_file, "w") as f: json.dump({"macro_halt": True}, f)
                        print("🛑 [MACRO HALT] High Impact USD Event! Halting Execution.")
                        await send_telegram_alert(session, "🛑 *تجميد الطوارئ (MACRO HALT):* خبر اقتصادي أمريكي عالي الخطورة الآن! تم إيقاف ذراع التنفيذ مؤقتاً لحمايتك من الانزلاق السعري.")
                    elif not halt_active and macro_halt:
                        macro_halt = False
                        with open(state_file, "w") as f: json.dump({"macro_halt": False}, f)
                        print("🟢 [MACRO CLEAR] Event passed. Execution resumed.")
                        await send_telegram_alert(session, "🟢 *زوال الخطر:* هدأ تأثير الخبر الاقتصادي، عودة ذراع التنفيذ للعمل. سيتم تنفيذ الصفقات المعلقة إن وُجدت.")
        except Exception:
            pass
            
        if time.time() - last_successful_macro_fetch > 7200: 
            if not macro_halt:
                macro_halt = True
                with open(state_file, "w") as f: json.dump({"macro_halt": True}, f)
                await send_telegram_alert(session, "⚠️ *عطل أمني (API Fail-Safe):* فشل الرادار الاقتصادي في الاتصال بالسيرفر لأكثر من ساعتين. تم فرض تجميد وقائي لمنع التداول العشوائي الأعمى.")
                
        await asyncio.sleep(60)

async def insider_scout(session):
    global agent_heartbeats
    print("🐋 [INSIDER] SEC Insider Trading Scout deployed.")
    while True:
        if bot_state != "RUNNING":
            await asyncio.sleep(10)
            continue
        agent_heartbeats["insider"] = time.time()
        try:
            if halal_tickers:
                top_tickers = list(halal_tickers)[:10]
                for t in top_tickers:
                    tk = yf.Ticker(t)
                    insider_tx = await asyncio.to_thread(lambda: tk.insider_transactions)
                    if insider_tx is not None and not insider_tx.empty:
                        latest = insider_tx.iloc[0]
                        shares = latest.get('Shares', 0)
                        if abs(shares) > 100000:
                            guid = f"insider_{t}_{latest.name}"
                            if guid not in seen_guids:
                                seen_guids.add(guid)
                                action = "شراء" if shares > 0 else "بيع"
                                item = {
                                    "source": "SEC Form 4 (Insider)",
                                    "title": f"حوت داخلي: {action} استباقي ضخم في سهم {t}",
                                    "summary": f"المدراء التنفيذيون لشركة {t} قاموا بـ {action} {abs(shares)} سهم مؤخراً.",
                                    "guid": guid
                                }
                                await raw_news_queue.put(item)
        except:
            pass
        await asyncio.sleep(300)

async def execution_agent(session):
    global agent_heartbeats, paper_trades, pending_execution_queue
    print("⚙️ [EXECUTION] Paper Trading Execution Arm deployed.")
    while True:
        if bot_state != "RUNNING":
            await asyncio.sleep(2)
            continue
            
        agent_heartbeats["execution"] = time.time()
        
        if not macro_halt and pending_execution_queue:
            to_execute = list(pending_execution_queue)
            pending_execution_queue.clear()
            for trade in to_execute:
                if time.time() - trade['timestamp'] < 3600:
                    trade['status'] = "تم التنفيذ (بعد زوال الخطر)"
                    paper_trades.insert(0, trade)
                    msg = f"""⚙️ *تنفيذ إشارة معلقة (محاكي)*
العملية: {trade['action']}
السهم: {trade['ticker']}
السعر التقريبي: ${trade['price']}"""
                    await send_telegram_alert(session, msg)
                else:
                    await send_telegram_alert(session, f"🗑️ *إلغاء إشارة معلقة:* تم إلغاء شراء {trade['ticker']} لمرور أكثر من ساعة عليها في غرفة الانتظار.")
            paper_trades = paper_trades[:20]

        try:
            trade_signal = await asyncio.wait_for(execution_queue.get(), timeout=2.0)
            
            trade = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "timestamp": time.time(),
                "ticker": trade_signal["ticker"],
                "action": trade_signal["action"],
                "price": trade_signal["price"],
                "status": "تم التنفيذ (محاكي)"
            }
            
            if macro_halt:
                print("⚠️ [EXECUTION] Signal routed to PENDING QUEUE due to MACRO HALT.")
                trade["status"] = "معلق في غرفة الانتظار (Macro Halt)"
                pending_execution_queue.append(trade)
                await send_telegram_alert(session, f"⏳ *غرفة الانتظار:* تم تحويل إشارة {trade['ticker']} لغرفة الانتظار بسبب الخطر الاقتصادي الحالي.")
            else:
                paper_trades.insert(0, trade)
                paper_trades = paper_trades[:20]
                msg = f"""⚙️ *تأكيد التنفيذ (محاكي)*
العملية: {trade['action']}
السهم: {trade['ticker']}
السعر التقريبي: ${trade['price']}"""
                await send_telegram_alert(session, msg)
                
            execution_queue.task_done()
        except asyncio.TimeoutError:
            pass

async def overseer_agent(session):
    print("👁️‍🗨️ [OVERSEER] Sentinel Health Monitor deployed.")
    while True:
        if bot_state == "RUNNING":
            now = time.time()
            for agent_name, last_beat in agent_heartbeats.items():
                if last_beat > 0 and (now - last_beat) > 300:
                    print(f"🚨 [OVERSEER] Agent {agent_name} is HUNG!")
                    await send_telegram_alert(session, f"🚨 *تحذير المفتش العام (Overseer):*\nالوكيل `{agent_name}` لا يستجيب منذ 5 دقائق! السرب يواجه اختناقاً.")
                    agent_heartbeats[agent_name] = now
        await asyncio.sleep(60)

def _save_json(payload):

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

async def run_swarm():
    print("="*60)
    print("🐝 THE NEURAL SWARM IS INITIALIZING (Omni-Control Enabled) 🐝")
    print("="*60)
    
    load_halal_stocks()
    _save_json({"last_updated": "جاري التهيئة...", "news": [], "telegram_status": "waiting", "bot_state": bot_state})
    
    await start_web_server()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for name, url in RSS_FEEDS.items():
            tasks.append(asyncio.create_task(rss_scout(session, name, url)))
        for sub in REDDIT_SUBS:
            tasks.append(asyncio.create_task(reddit_scout(session, sub)))
            
        tasks.append(asyncio.create_task(domino_agent()))
        tasks.append(asyncio.create_task(macro_agent(session)))
        tasks.append(asyncio.create_task(insider_scout(session)))
        tasks.append(asyncio.create_task(execution_agent(session)))
        tasks.append(asyncio.create_task(overseer_agent(session)))
        tasks.append(asyncio.create_task(filter_agent()))
        tasks.append(asyncio.create_task(master_agent(session)))
        tasks.append(asyncio.create_task(telegram_listener(session)))
        tasks.append(asyncio.create_task(smart_schedule()))
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_swarm())
    except KeyboardInterrupt:
        pass
