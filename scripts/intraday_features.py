"""
intraday_features.py — Intraday Feature Engineering for Day Trading
====================================================================
Computes features from 5-minute bars for intraday trading signals.

Key features: VWAP deviation, Opening Range Breakout, Relative Volume,
Intraday momentum, RSI/MACD on 5-min bars, Session context.

Usage:
    from intraday_features import IntradayFeatureEngine
    engine = IntradayFeatureEngine(alpaca_client)
    features = engine.build_features("AAPL")
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

log = logging.getLogger("IntradayFeatures")
ET = ZoneInfo("America/New_York")

WARMUP_DAYS = 5
MIN_BARS_5M = 78  # 1 full trading day of 5-min bars


class IntradayFeatureEngine:
    """Builds intraday features for a single ticker from 5-min bars."""

    def __init__(self, alpaca_client):
        self.alpaca = alpaca_client

    def _get_5min_bars(self, ticker, lookback_days=WARMUP_DAYS):
        """Fetch recent 5-minute bars from Alpaca."""
        end = datetime.now(ET)
        start = end - timedelta(days=lookback_days + 2)
        start_str = start.strftime("%Y-%m-%dT09:30:00Z")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            url = (f"{self.alpaca.DATA_URL}/v2/stocks/{ticker}/bars"
                   f"?timeframe=5Min&start={start_str}&end={end_str}"
                   f"&limit=10000&feed=iex&adjustment=raw")
            data = self.alpaca._get(url)
            bars = data.get("bars", [])
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df = df.rename(columns={"t": "datetime", "o": "open", "h": "high",
                                    "l": "low", "c": "close", "v": "volume",
                                    "vw": "vwap"})
            df = df.set_index("datetime")[["open", "high", "low", "close",
                                           "volume", "vwap"]]
            df.index = df.index.tz_localize(None)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            log.warning(f"  5-min bars failed for {ticker}: {e}")
            return pd.DataFrame()

    def _compute_vwap_features(self, df_today):
        """VWAP deviation — the #1 intraday signal."""
        if df_today.empty or len(df_today) < 5:
            return {"vwap_deviation": 0.0, "vwap_slope": 0.0, "vwap_cross_bars_ago": 0}
        close = df_today["close"]
        vwap = df_today["vwap"] if "vwap" in df_today.columns else close
        dev = (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
        slope = (vwap.iloc[-1] - vwap.iloc[-6]) / vwap.iloc[-6] if len(vwap) >= 6 else 0.0
        cross_mask = (close > vwap).astype(int).diff().abs()
        last_cross = cross_mask[cross_mask == 1].index
        bars_since = len(df_today) - df_today.index.get_loc(last_cross[-1]) if len(last_cross) > 0 else len(df_today)
        return {"vwap_deviation": round(float(dev), 6), "vwap_slope": round(float(slope), 6),
                "vwap_cross_bars_ago": int(bars_since)}

    def _compute_opening_range(self, df_today):
        """Opening Range Breakout — first 15 min high/low."""
        if df_today.empty or len(df_today) < 3:
            return {"orb_position": 0.0, "orb_width_pct": 0.0, "orb_breakout": 0}
        first_15 = df_today.iloc[:3]
        or_high = first_15["high"].max()
        or_low = first_15["low"].min()
        or_width = (or_high - or_low) / or_low if or_low > 0 else 0
        price = df_today["close"].iloc[-1]
        orb_pos = (price - or_low) / (or_high - or_low) * 2 - 1 if or_high != or_low else 0.0
        breakout = 1 if price > or_high else (-1 if price < or_low else 0)
        return {"orb_position": round(float(np.clip(orb_pos, -2, 2)), 4),
                "orb_width_pct": round(float(or_width), 6), "orb_breakout": int(breakout)}

    def _compute_relative_volume(self, df_5m, df_today):
        """Current volume vs average — detects institutional interest."""
        if df_today.empty or df_5m.empty:
            return {"relative_volume": 1.0, "volume_surge": 0}
        current_vol = df_today["volume"].iloc[-1]
        recent_avg = df_5m["volume"].tail(78 * 3).mean()
        if recent_avg <= 0:
            recent_avg = 1
        rel_vol = current_vol / recent_avg
        surge = 0
        if len(df_today) >= 6:
            surge = 1 if df_today["volume"].tail(6).mean() > (recent_avg * 2) else 0
        return {"relative_volume": round(float(np.clip(rel_vol, 0, 10)), 4),
                "volume_surge": int(surge)}

    def _compute_momentum(self, df_5m):
        """Multi-timeframe intraday momentum on 5-min bars."""
        if len(df_5m) < 36:
            return {"momentum_6bar": 0.0, "momentum_12bar": 0.0,
                    "momentum_36bar": 0.0, "momentum_accel": 0.0}
        close = df_5m["close"]
        m6 = close.pct_change(6).iloc[-1]
        m12 = close.pct_change(12).iloc[-1]
        m36 = close.pct_change(36).iloc[-1]
        m12_prev = close.pct_change(12).iloc[-7] if len(close) >= 19 else 0
        accel = m12 - m12_prev
        def safe(v): return round(float(v if not np.isnan(v) else 0), 6)
        return {"momentum_6bar": safe(m6), "momentum_12bar": safe(m12),
                "momentum_36bar": safe(m36), "momentum_accel": safe(accel)}

    def _compute_technicals(self, df_5m):
        """RSI, MACD, Stoch, ADX, BB, EMA, ATR on 5-min bars."""
        defaults = {"rsi_14": 50.0, "macd_hist": 0.0, "stoch_k": 50.0,
                     "adx_14": 20.0, "bb_position": 0.0, "ema_9_dist": 0.0,
                     "ema_21_dist": 0.0, "atr_pct": 0.02}
        if len(df_5m) < 30:
            return defaults
        c, h, lo = df_5m["close"], df_5m["high"], df_5m["low"]
        feat = {}
        rsi = ta.rsi(c, length=14)
        feat["rsi_14"] = round(float(rsi.iloc[-1] if rsi is not None and not rsi.empty else 50), 2)
        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None and "MACDh_12_26_9" in macd.columns:
            v = macd["MACDh_12_26_9"].iloc[-1]
            feat["macd_hist"] = round(float(v / c.iloc[-1] if not np.isnan(v) else 0), 6)
        else:
            feat["macd_hist"] = 0.0
        stoch = ta.stoch(h, lo, c)
        feat["stoch_k"] = round(float(stoch["STOCHk_14_3_3"].iloc[-1] or 50), 2) if stoch is not None and "STOCHk_14_3_3" in stoch.columns else 50.0
        adx_df = ta.adx(h, lo, c, length=14)
        feat["adx_14"] = round(float(adx_df["ADX_14"].iloc[-1] or 20), 2) if adx_df is not None and "ADX_14" in adx_df.columns else 20.0
        bb = ta.bbands(c, length=20, std=2)
        if bb is not None:
            bbu = [x for x in bb.columns if "BBU" in x]
            bbl = [x for x in bb.columns if "BBL" in x]
            if bbu and bbl:
                u, l = bb[bbu[0]].iloc[-1], bb[bbl[0]].iloc[-1]
                feat["bb_position"] = round(float((c.iloc[-1] - l) / (u - l) * 2 - 1 if u != l else 0), 4)
            else:
                feat["bb_position"] = 0.0
        else:
            feat["bb_position"] = 0.0
        ema9 = ta.ema(c, length=9)
        ema21 = ta.ema(c, length=21)
        feat["ema_9_dist"] = round(float((c.iloc[-1] - ema9.iloc[-1]) / c.iloc[-1]), 6) if ema9 is not None and not ema9.empty else 0.0
        feat["ema_21_dist"] = round(float((c.iloc[-1] - ema21.iloc[-1]) / c.iloc[-1]), 6) if ema21 is not None and not ema21.empty else 0.0
        atr = ta.atr(h, lo, c, length=14)
        feat["atr_pct"] = round(float(atr.iloc[-1] / c.iloc[-1] if atr is not None and not atr.empty and not np.isnan(atr.iloc[-1]) else 0.02), 6)
        return feat

    def _session_context(self):
        """Time-of-day features (opening, midday, closing)."""
        now = datetime.now(ET)
        open_t = now.replace(hour=9, minute=30, second=0)
        close_t = now.replace(hour=16, minute=0, second=0)
        mins_open = max(0, (now - open_t).total_seconds() / 60)
        mins_close = max(0, (close_t - now).total_seconds() / 60)
        pct = min(1.0, mins_open / 390)
        return {"minutes_since_open": round(mins_open, 1), "minutes_to_close": round(mins_close, 1),
                "session_pct": round(pct, 4), "is_opening": 1 if mins_open < 30 else 0,
                "is_midday": 1 if 120 < mins_open < 270 else 0,
                "is_closing": 1 if mins_close < 30 else 0}

    def _intraday_volatility(self, df_5m):
        """Volatility metrics on 5-min bars."""
        if len(df_5m) < 20:
            return {"vol_20bar": 0.02, "vol_6bar": 0.02, "vol_ratio": 1.0, "hl_range_pct": 0.01}
        lr = np.log(df_5m["close"] / df_5m["close"].shift(1))
        v20 = lr.tail(20).std()
        v6 = lr.tail(6).std()
        vr = v6 / v20 if v20 > 0 else 1.0
        th = df_5m["high"].tail(78).max()
        tl = df_5m["low"].tail(78).min()
        hlr = (th - tl) / tl if tl > 0 else 0
        def sf(v, d=0.02): return round(float(v if not np.isnan(v) else d), 6)
        return {"vol_20bar": sf(v20), "vol_6bar": sf(v6),
                "vol_ratio": round(float(np.clip(vr, 0, 5) if not np.isnan(vr) else 1.0), 4),
                "hl_range_pct": sf(hlr, 0.01)}

    def build_features(self, ticker):
        """Build complete intraday feature set. Returns dict or None."""
        df_5m = self._get_5min_bars(ticker)
        if df_5m.empty or len(df_5m) < MIN_BARS_5M:
            return None
        now = datetime.now(ET)
        today_str = now.strftime("%Y-%m-%d")
        df_today = df_5m[df_5m.index.strftime("%Y-%m-%d") == today_str]
        if df_today.empty or len(df_today) < 3:
            return None
        features = {"price": float(df_5m["close"].iloc[-1])}
        features.update(self._compute_vwap_features(df_today))
        features.update(self._compute_opening_range(df_today))
        features.update(self._compute_relative_volume(df_5m, df_today))
        features.update(self._compute_momentum(df_5m))
        features.update(self._compute_technicals(df_5m))
        features.update(self._session_context())
        features.update(self._intraday_volatility(df_5m))
        return features

    def compute_entry_score(self, features, catalyst_score=0.0):
        """
        Rule-based entry scoring [0, 100]. Used until intraday model is trained.
        Breakdown: VWAP(25) + Momentum(20) + Volume(15) + ORB(15) + Tech(15) + News(10)
        Minimum to enter: 55
        """
        if features is None:
            return 0.0
        score = 0.0
        # VWAP alignment (0-25)
        vd = features.get("vwap_deviation", 0)
        vs = features.get("vwap_slope", 0)
        if vd > 0: score += min(15, vd * 1500)
        if vs > 0: score += min(10, vs * 5000)
        # Momentum (0-20)
        m12 = features.get("momentum_12bar", 0)
        ma = features.get("momentum_accel", 0)
        if m12 > 0: score += min(12, m12 * 800)
        if ma > 0: score += min(8, ma * 2000)
        # Volume (0-15)
        rv = features.get("relative_volume", 1.0)
        if rv > 1.5: score += min(10, (rv - 1) * 10)
        if features.get("volume_surge", 0): score += 5
        # ORB (0-15)
        if features.get("orb_breakout", 0) == 1: score += 10
        op = features.get("orb_position", 0)
        if op > 0.5: score += min(5, op * 5)
        # Technicals (0-15)
        rsi = features.get("rsi_14", 50)
        if 40 < rsi < 70: score += 5
        if features.get("macd_hist", 0) > 0: score += 4
        if features.get("ema_9_dist", 0) > 0: score += 3
        bp = features.get("bb_position", 0)
        if -0.5 < bp < 0.7: score += 3
        # News catalyst (0-10)
        score += min(10, catalyst_score * 10)
        # Session penalties
        if features.get("is_midday", 0): score *= 0.85
        if features.get("minutes_to_close", 390) < 30: score *= 0.5
        return round(min(100, max(0, score)), 1)
