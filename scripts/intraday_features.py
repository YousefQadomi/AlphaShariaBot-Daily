"""
intraday_features.py — Intraday Feature Engineering for Day Trading
====================================================================
Computes features from 5-minute bars for intraday trading signals.

Key features: VWAP deviation, Opening Range Breakout, Relative Volume,
Intraday momentum, RSI/MACD on 5-min bars, Session context,
Gap detection, Microstructure analysis.

Usage:
    from intraday_features import IntradayFeatureEngine
    engine = IntradayFeatureEngine(alpaca_client)
    features = engine.build_features("AAPL")
"""

import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
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

        # Retry with backoff for 429 rate limit errors
        import time as _time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = (f"{self.alpaca.DATA_URL}/v2/stocks/{ticker}/bars"
                       f"?timeframe=5Min&start={start_str}&end={end_str}"
                       f"&limit=10000&feed=sip&adjustment=raw")
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
                # Convert UTC -> ET before stripping timezone
                # (fixes bug where today's bars were missed due to UTC dates)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(ET).tz_localize(None)
                else:
                    df.index = df.index.tz_localize(None)
                df.sort_index(inplace=True)
                return df
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # 2s, 4s
                    log.debug(f"  Rate limited on {ticker}, retrying in {wait}s...")
                    _time.sleep(wait)
                    continue
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
        """RSI, MACD, Stoch, ADX, BB, EMA, ATR on 5-min bars.
        Optimized: pre-slice close/high/low once, use try/except blocks."""
        defaults = {"rsi_14": 50.0, "macd_hist": 0.0, "macd_hist_prev": 0.0,
                     "stoch_k": 50.0, "adx_14": 20.0, "bb_position": 0.0,
                     "ema_9_dist": 0.0, "ema_21_dist": 0.0,
                     "ema_9_above_21": 0, "atr_pct": 0.02}
        if len(df_5m) < 30:
            return defaults

        # Pre-compute slices once (optimization: avoid repeated column access)
        c = df_5m["close"]
        h = df_5m["high"]
        lo = df_5m["low"]
        last_close = c.iloc[-1]
        feat = {}

        # RSI
        try:
            rsi = ta.rsi(c, length=14)
            feat["rsi_14"] = round(float(rsi.iloc[-1]), 2)
        except Exception:
            feat["rsi_14"] = 50.0

        # MACD (current + previous histogram for trend detection)
        try:
            macd = ta.macd(c, fast=12, slow=26, signal=9)
            hist_col = "MACDh_12_26_9"
            v = macd[hist_col].iloc[-1]
            feat["macd_hist"] = round(float(v / last_close if not np.isnan(v) else 0), 6)
            v_prev = macd[hist_col].iloc[-2] if len(macd) >= 2 else 0
            feat["macd_hist_prev"] = round(float(v_prev / last_close if not np.isnan(v_prev) else 0), 6)
        except Exception:
            feat["macd_hist"] = 0.0
            feat["macd_hist_prev"] = 0.0

        # Stochastic
        try:
            stoch = ta.stoch(h, lo, c)
            feat["stoch_k"] = round(float(stoch["STOCHk_14_3_3"].iloc[-1] or 50), 2)
        except Exception:
            feat["stoch_k"] = 50.0

        # ADX (cached via single computation)
        try:
            adx_df = ta.adx(h, lo, c, length=14)
            feat["adx_14"] = round(float(adx_df["ADX_14"].iloc[-1] or 20), 2)
        except Exception:
            feat["adx_14"] = 20.0

        # Bollinger Bands
        try:
            bb = ta.bbands(c, length=20, std=2)
            bbu = [x for x in bb.columns if "BBU" in x]
            bbl = [x for x in bb.columns if "BBL" in x]
            u, l = bb[bbu[0]].iloc[-1], bb[bbl[0]].iloc[-1]
            feat["bb_position"] = round(float((last_close - l) / (u - l) * 2 - 1 if u != l else 0), 4)
        except Exception:
            feat["bb_position"] = 0.0

        # EMAs (9 and 21) + crossover detection
        try:
            ema9 = ta.ema(c, length=9)
            ema21 = ta.ema(c, length=21)
            ema9_val = ema9.iloc[-1]
            ema21_val = ema21.iloc[-1]
            feat["ema_9_dist"] = round(float((last_close - ema9_val) / last_close), 6)
            feat["ema_21_dist"] = round(float((last_close - ema21_val) / last_close), 6)
            feat["ema_9_above_21"] = 1 if ema9_val > ema21_val else 0
        except Exception:
            feat["ema_9_dist"] = 0.0
            feat["ema_21_dist"] = 0.0
            feat["ema_9_above_21"] = 0

        # ATR
        try:
            atr = ta.atr(h, lo, c, length=14)
            atr_val = atr.iloc[-1]
            feat["atr_pct"] = round(float(atr_val / last_close if not np.isnan(atr_val) else 0.02), 6)
        except Exception:
            feat["atr_pct"] = 0.02

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

    def _compute_gap_features(self, df_5m, df_today):
        """Detect pre-market gap and gap continuation.

        Compares today's open to the previous trading day's close to detect
        gap-ups/downs and whether price is continuing or filling the gap.

        Returns:
            dict with gap_pct, gap_fill_pct, gap_continuation
        """
        defaults = {"gap_pct": 0.0, "gap_fill_pct": 0.0, "gap_continuation": 0}
        if df_today.empty or len(df_today) < 3 or len(df_5m) < MIN_BARS_5M + 3:
            return defaults

        try:
            today_date = df_today.index[0].date()
            # Get previous day's bars (all bars before today)
            prev_days = df_5m[df_5m.index.date < today_date]
            if prev_days.empty:
                return defaults

            yesterday_close = prev_days["close"].iloc[-1]
            today_open = df_today["open"].iloc[0]
            current_price = df_today["close"].iloc[-1]

            if yesterday_close <= 0:
                return defaults

            # Gap percentage: positive = gap up, negative = gap down
            gap_pct = (today_open - yesterday_close) / yesterday_close

            # Gap fill percentage: 0 = no fill, 1 = fully filled, >1 = overfilled
            if abs(gap_pct) < 0.001:
                # Negligible gap
                gap_fill_pct = 0.0
            else:
                gap_size = today_open - yesterday_close
                fill_amount = today_open - current_price  # how much has reverted
                gap_fill_pct = fill_amount / gap_size if gap_size != 0 else 0.0
                gap_fill_pct = float(np.clip(gap_fill_pct, -0.5, 2.0))

            # Gap continuation: +1 if price moving further in gap direction
            #                   -1 if price reversing (filling gap)
            #                    0 if no significant gap
            if abs(gap_pct) < 0.002:
                gap_continuation = 0
            elif gap_pct > 0:
                # Gap up: continuation if price above open, reversal if below
                gap_continuation = 1 if current_price > today_open else -1
            else:
                # Gap down: continuation if price below open, reversal if above
                gap_continuation = 1 if current_price < today_open else -1

            return {
                "gap_pct": round(float(gap_pct), 6),
                "gap_fill_pct": round(float(gap_fill_pct), 4),
                "gap_continuation": int(gap_continuation),
            }
        except Exception:
            return defaults

    def _compute_microstructure(self, df_today):
        """Intrabar price action features.

        Analyzes bar-level price action to detect buying/selling pressure:
        - bar_range_avg: average (high-low)/close of recent bars (volatility proxy)
        - close_position_in_bar: where close sits in each bar's range (0=low, 1=high)
          Consistently closing near highs = buying pressure
        - consecutive_up_bars: count of consecutive bars where close > open
          (momentum continuation signal)
        """
        defaults = {"bar_range_avg": 0.0, "close_position_in_bar": 0.5,
                     "consecutive_up_bars": 0}
        if df_today.empty or len(df_today) < 6:
            return defaults

        try:
            recent = df_today.tail(12)  # last ~1 hour of bars
            highs = recent["high"].values
            lows = recent["low"].values
            closes = recent["close"].values
            opens = recent["open"].values

            # Average bar range as percentage of close
            bar_ranges = (highs - lows) / np.where(closes > 0, closes, 1.0)
            bar_range_avg = float(np.nanmean(bar_ranges))

            # Where does close sit within each bar's range? (0 = at low, 1 = at high)
            bar_widths = highs - lows
            close_positions = np.where(
                bar_widths > 0,
                (closes - lows) / bar_widths,
                0.5  # doji bar
            )
            # Weight recent bars more heavily (exponential decay)
            n = len(close_positions)
            weights = np.exp(np.linspace(-1.0, 0.0, n))
            weights /= weights.sum()
            close_position_avg = float(np.average(close_positions, weights=weights))

            # Count consecutive up bars from the most recent bar backwards
            consecutive_up = 0
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] > opens[i]:
                    consecutive_up += 1
                else:
                    break

            return {
                "bar_range_avg": round(bar_range_avg, 6),
                "close_position_in_bar": round(close_position_avg, 4),
                "consecutive_up_bars": int(consecutive_up),
            }
        except Exception:
            return defaults

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
        features.update(self._compute_gap_features(df_5m, df_today))
        features.update(self._compute_microstructure(df_today))
        # Ensure atr_pct is always in output (already set by _compute_technicals)
        if "atr_pct" not in features:
            features["atr_pct"] = 0.02
        return features

    def compute_entry_score(self, features, catalyst_score=0.0):
        """
        Improved rule-based entry scoring [0, 100].
        Breakdown:
          VWAP alignment:    0-20 (above VWAP with positive slope = bullish)
          Momentum:          0-20 (multi-timeframe agreement)
          Volume:            0-15 (relative volume + surge detection)
          ORB:               0-10 (opening range breakout)
          Technical:         0-15 (RSI, MACD, EMA alignment, BB position)
          News catalyst:     0-10 (from news fetcher)
          Gap analysis:      0-10 (pre-market gap continuation)

        Session adjustments:
          - Opening 30 min: +5 bonus (best opportunity window)
          - Power hour (3-3:45 PM): +3 bonus
          - Midday (11:30-2:00): -5 penalty (low volume, choppy)
          - Last 15 min: score = 0 (no new entries)

        Returns:
            (score, details_dict) where details_dict includes component scores
            and atr_pct for position sizing.
        """
        empty_details = {
            "vwap_score": 0, "momentum_score": 0, "volume_score": 0,
            "orb_score": 0, "technical_score": 0, "catalyst_score_out": 0,
            "gap_score": 0, "session_adj": 0, "atr_pct": 0.02,
        }
        if features is None:
            return (0.0, empty_details)

        # ── VWAP Alignment (0-20) ──────────────────────────────────────
        vwap_score = 0.0
        vd = features.get("vwap_deviation", 0)
        vs = features.get("vwap_slope", 0)
        if vd > 0:
            # Above VWAP: scale up to 12 pts (stronger for larger deviation)
            vwap_score += min(12, vd * 1200)
        else:
            # Below VWAP: penalty (not zero — being below VWAP is bearish)
            vwap_score += max(-5, vd * 500)

        if vs > 0:
            # VWAP slope positive = uptrend confirmation
            vwap_score += min(8, vs * 4000)
        else:
            # Negative slope = trend weakening
            vwap_score += max(-3, vs * 1500)
        vwap_score = max(0, min(20, vwap_score))

        # ── Momentum — Multi-timeframe Agreement (0-20) ───────────────
        momentum_score = 0.0
        m6 = features.get("momentum_6bar", 0)
        m12 = features.get("momentum_12bar", 0)
        m36 = features.get("momentum_36bar", 0)
        ma = features.get("momentum_accel", 0)

        # Both short-term timeframes must agree (noise reduction)
        if m6 > 0 and m12 > 0:
            # Multi-timeframe agreement: strong signal
            agreement_strength = min(m6, m12)  # limited by weaker signal
            momentum_score += min(12, agreement_strength * 1000)
            # Bonus if 36-bar also agrees (triple alignment)
            if m36 > 0:
                momentum_score += min(4, m36 * 200)
        elif m6 > 0 or m12 > 0:
            # Single timeframe only: weak signal (noise)
            single_m = max(m6, m12)
            momentum_score += min(4, single_m * 300)

        # Acceleration bonus (momentum increasing)
        if ma > 0:
            momentum_score += min(4, ma * 1500)
        momentum_score = max(0, min(20, momentum_score))

        # ── Volume (0-15) ─────────────────────────────────────────────
        volume_score = 0.0
        rv = features.get("relative_volume", 1.0)
        if rv > 2.0:
            # Strong relative volume: very significant
            volume_score += min(10, (rv - 1.0) * 5)
        elif rv > 1.5:
            # Moderate: somewhat significant
            volume_score += min(6, (rv - 1.0) * 6)
        elif rv > 1.2:
            # Slightly above average
            volume_score += min(3, (rv - 1.0) * 10)

        # Volume surge (sustained elevated volume) weighted higher
        if features.get("volume_surge", 0):
            volume_score += 7
        volume_score = max(0, min(15, volume_score))

        # ── ORB — Opening Range Breakout (0-10) ──────────────────────
        orb_score = 0.0
        if features.get("orb_breakout", 0) == 1:
            orb_score += 7
        op = features.get("orb_position", 0)
        if op > 0.5:
            orb_score += min(3, op * 2.5)
        orb_score = max(0, min(10, orb_score))

        # ── Technicals (0-15) ────────────────────────────────────────
        tech_score = 0.0

        # RSI sweet spot: 45-65 is ideal for momentum entries
        rsi = features.get("rsi_14", 50)
        if 45 <= rsi <= 65:
            # Sweet spot: full points
            tech_score += 4
        elif 35 <= rsi < 45:
            # Slightly oversold, could bounce
            tech_score += 2
        elif 65 < rsi <= 75:
            # Getting overbought, reduced score
            tech_score += 1
        # RSI > 75 or < 35: 0 points (extremes are risky for new entries)

        # MACD histogram: positive AND increasing = strong
        macd_h = features.get("macd_hist", 0)
        macd_h_prev = features.get("macd_hist_prev", 0)
        if macd_h > 0:
            tech_score += 2
            if macd_h > macd_h_prev:
                # Histogram increasing = momentum accelerating
                tech_score += 2
        elif macd_h < 0 and macd_h > macd_h_prev:
            # Negative but improving = potential reversal, small credit
            tech_score += 1

        # EMA crossover: EMA 9 > EMA 21 = bullish structure
        if features.get("ema_9_above_21", 0):
            tech_score += 3
        # Price above EMA 9 = immediate trend confirmation
        if features.get("ema_9_dist", 0) > 0:
            tech_score += 1

        # Bollinger Band position: near middle-upper is ideal
        bp = features.get("bb_position", 0)
        if 0.0 < bp < 0.6:
            tech_score += 2
        elif -0.3 < bp <= 0.0:
            tech_score += 1
        tech_score = max(0, min(15, tech_score))

        # ── News Catalyst (0-10) ─────────────────────────────────────
        cat_score = min(10, catalyst_score * 10)

        # ── Gap Analysis (0-10) ──────────────────────────────────────
        gap_score = 0.0
        gap_pct = features.get("gap_pct", 0)
        gap_cont = features.get("gap_continuation", 0)
        gap_fill = features.get("gap_fill_pct", 0)

        if gap_pct > 0.01 and gap_cont == 1:
            # Gap up with continuation = strong bullish
            gap_score += min(7, gap_pct * 300)
            # Less fill = stronger continuation
            if gap_fill < 0.3:
                gap_score += 3
        elif gap_pct > 0.005 and gap_cont == 1:
            # Small gap up with continuation
            gap_score += min(4, gap_pct * 200)
        elif gap_pct < -0.01 and gap_cont == -1:
            # Gap down reversing (filling up) = potential long
            gap_score += min(4, abs(gap_pct) * 150)
        gap_score = max(0, min(10, gap_score))

        # ── Raw score (pre-session) ──────────────────────────────────
        raw_score = (vwap_score + momentum_score + volume_score +
                     orb_score + tech_score + cat_score + gap_score)

        # ── Session Adjustments ──────────────────────────────────────
        session_adj = 0.0
        mins_open = features.get("minutes_since_open", 195)
        mins_close = features.get("minutes_to_close", 195)

        # Last 15 minutes: NO new entries (forced zero)
        if mins_close < 15:
            details = {
                "vwap_score": round(vwap_score, 1),
                "momentum_score": round(momentum_score, 1),
                "volume_score": round(volume_score, 1),
                "orb_score": round(orb_score, 1),
                "technical_score": round(tech_score, 1),
                "catalyst_score_out": round(cat_score, 1),
                "gap_score": round(gap_score, 1),
                "session_adj": -999,
                "atr_pct": features.get("atr_pct", 0.02),
            }
            return (0.0, details)

        # Opening 30 minutes: best opportunity window
        if mins_open < 30:
            session_adj += 5
        # Power hour: 3:00-3:45 PM (330-375 mins since open)
        elif 330 <= mins_open <= 375:
            session_adj += 3
        # Midday lull: 11:30 AM - 2:00 PM (120-270 mins since open)
        elif 120 < mins_open < 270:
            session_adj -= 5

        final_score = round(min(100, max(0, raw_score + session_adj)), 1)

        details = {
            "vwap_score": round(vwap_score, 1),
            "momentum_score": round(momentum_score, 1),
            "volume_score": round(volume_score, 1),
            "orb_score": round(orb_score, 1),
            "technical_score": round(tech_score, 1),
            "catalyst_score_out": round(cat_score, 1),
            "gap_score": round(gap_score, 1),
            "session_adj": round(session_adj, 1),
            "atr_pct": features.get("atr_pct", 0.02),
        }

        return (final_score, details)
