# -*- coding: utf-8 -*-
"""Technical indicators, 200MA breakout states and scoring for Quant Screener V4.5."""
from __future__ import annotations
import numpy as np
import pandas as pd

MA_WINDOWS = (5, 10, 15, 20, 60, 200)

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def _normalize_taipei_date(value):
    """Convert a market timestamp to a timezone-naive Taipei date."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Taipei").tz_localize(None)
    return ts.normalize()


def compute_indicators(df: pd.DataFrame, live_price: float | None = None, live_timestamp=None) -> dict | None:
    """
    Compute indicators using a strict 200-observation SMA.

    Price convention: Yahoo Finance Close (not Adj Close). Sort and deduplicate
    daily bars, ignore rows without a valid Close, and only substitute an
    intraday quote when its timestamp belongs to today's Taipei date.
    """
    if df is None or df.empty:
        return None

    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "close" not in x.columns:
        return None
    if "volume" not in x.columns:
        x["volume"] = 0

    x["close"] = _num(x["close"])
    x["volume"] = _num(x["volume"]).fillna(0)

    # The index must represent trading dates. Sort and keep one bar per date,
    # so duplicate/out-of-order rows cannot shift the SMA window.
    try:
        x.index = pd.DatetimeIndex(pd.to_datetime(x.index))
    except Exception:
        return None
    if x.index.tz is not None:
        x.index = x.index.tz_convert("Asia/Taipei").tz_localize(None)
    x = x.sort_index()
    x = x.loc[~x.index.duplicated(keep="last")]
    x = x.dropna(subset=["close"])
    if x.empty:
        return None

    # Only raw Close is used. Adj Close is deliberately ignored.
    close = x["close"].astype(float).copy()
    volume = x["volume"].astype(float).copy()

    today = pd.Timestamp.now(tz="Asia/Taipei").normalize().tz_localize(None)
    use_live_today = False
    if live_price is not None and live_timestamp is not None:
        try:
            live_day = _normalize_taipei_date(live_timestamp)
            live_value = float(live_price)
            use_live_today = live_day == today and np.isfinite(live_value) and live_value > 0
        except (TypeError, ValueError, OverflowError):
            use_live_today = False

    # Build one close per actual available trading bar. If today's daily bar
    # exists, replace its Close with the latest intraday quote; otherwise add
    # today's quote once. Never append an old/stale quote or a weekend quote.
    analysis = pd.DataFrame({"close": close, "volume": volume})
    if use_live_today:
        today_rows = analysis.index.normalize() == today
        if today_rows.any():
            analysis.loc[analysis.index[today_rows][-1], "close"] = live_value
        else:
            analysis.loc[today] = {"close": live_value, "volume": 0.0}
            analysis = analysis.sort_index()

    analysis = analysis.dropna(subset=["close"])
    if len(analysis) < 200:
        return None

    # 200MA is the simple average of exactly the latest 200 valid trading bars.
    # Do not use min_periods fallbacks or Adj Close.
    close_series = analysis["close"].astype(float)
    ma = {n: close_series.rolling(window=n, min_periods=n).mean() for n in MA_WINDOWS}
    last_ma = {n: float(series.iloc[-1]) for n, series in ma.items()}
    ma200 = ma[200]
    ma200_now = last_ma[200]

    ma200_5d = float(ma200.iloc[-6]) if len(ma200) >= 206 and pd.notna(ma200.iloc[-6]) else np.nan
    ma200_20d = float(ma200.iloc[-21]) if len(ma200) >= 221 and pd.notna(ma200.iloc[-21]) else np.nan
    slope20 = (ma200_now / ma200_20d - 1.0) if pd.notna(ma200_20d) and ma200_20d else 0.0
    slope5 = (ma200_now / ma200_5d - 1.0) if pd.notna(ma200_5d) and ma200_5d else 0.0

    c = float(close_series.iloc[-1])
    prev_c = float(close_series.iloc[-2]) if len(close_series) >= 2 else np.nan
    prev_ma200 = float(ma200.iloc[-2]) if len(ma200) >= 2 and pd.notna(ma200.iloc[-2]) else np.nan
    crossed_up = pd.notna(prev_c) and pd.notna(prev_ma200) and prev_c <= prev_ma200 and c > ma200_now
    crossed_down = pd.notna(prev_c) and pd.notna(prev_ma200) and prev_c >= prev_ma200 and c < ma200_now

    above = close_series > ma200
    current_above = bool(above.iloc[-1])
    run = 0
    for v in reversed(above.tolist()):
        if bool(v) == current_above:
            run += 1
        else:
            break

    vol5 = float(analysis["volume"].tail(5).mean())
    vol20 = float(analysis["volume"].tail(20).mean())
    volume_ratio = vol5 / vol20 if vol20 else 0.0

    ma20 = last_ma[20]
    std20 = float(close_series.rolling(20, min_periods=20).std().iloc[-1])
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    width = (upper - lower) / ma20 if ma20 else np.nan
    prev20high = float(close_series.iloc[-21:-1].max()) if len(close_series) >= 21 else np.nan

    # A valid fresh breakout is a close/quote crossing its contemporaneous SMA.
    # The breakout session is day 1; retain only the latest 5 trading bars.
    recent_cross = False
    cross_days_ago = None
    breakout_day_number = None
    max_lookback = min(5, len(close_series) - 1)
    for i in range(1, max_lookback + 1):
        prev_close_i = close_series.iloc[-i-1]
        curr_close_i = close_series.iloc[-i]
        prev_ma_i = ma200.iloc[-i-1]
        curr_ma_i = ma200.iloc[-i]
        if pd.notna(prev_ma_i) and pd.notna(curr_ma_i) and prev_close_i <= prev_ma_i and curr_close_i > curr_ma_i:
            recent_cross = True
            cross_days_ago = i - 1
            breakout_day_number = i
            break

    recent_retest = False
    if c > ma200_now and len(close_series) >= 6:
        for i in range(1, min(6, len(close_series) - 1)):
            if close_series.iloc[-i] <= ma200.iloc[-i] * 1.01:
                recent_retest = True
                break

    if crossed_up:
        breakout_type = "今日突破200MA"
    elif recent_cross:
        breakout_type = f"近5日突破200MA（第{breakout_day_number}個交易日）"
    elif recent_retest and run >= 1:
        breakout_type = "200MA回踩後站回"
    elif current_above:
        breakout_type = "站上200MA（非新突破）"
    else:
        breakout_type = "200MA下方"

    return {
        "close": c, "prev_close": prev_c,
        "ma5": last_ma[5], "ma10": last_ma[10], "ma15": last_ma[15], "ma20": last_ma[20],
        "ma60": last_ma[60], "ma200": ma200_now,
        "ma200_slope5": float(slope5), "ma200_slope20": float(slope20),
        "bias20": float((c / ma20 - 1) * 100),
        "vol5": vol5, "vol20": vol20, "volume_ratio": float(volume_ratio),
        "bb_upper": float(upper), "bb_lower": float(lower),
        "bb_width": float(width) if pd.notna(width) else 0.0,
        "prev20high": prev20high, "above200": current_above,
        "above200_run": run, "crossed_up_200": bool(crossed_up), "crossed_down_200": bool(crossed_down),
        "recent_200_breakout": bool(recent_cross), "cross_days_ago": cross_days_ago,
        "breakout_day_number": breakout_day_number,
        "recent_200_retest": bool(recent_retest), "breakout_type": breakout_type,
    }

def strategy_flags(x: dict) -> dict:
    c=x["close"]; m5=x["ma5"]; m10=x["ma10"]; m20=x["ma20"]; m60=x["ma60"]; m200=x["ma200"]
    flags={}
    if x["above200"] and x["ma200_slope20"] > 0:
        flags["200MA 多頭趨勢"] = True
    if x["crossed_up_200"]:
        flags["200MA 今日突破"] = True
        if x["volume_ratio"] >= 1.5:
            flags["200MA 帶量突破"] = True
    if x["recent_200_breakout"] and x["volume_ratio"] >= 1.2:
        flags["200MA 近5日突破+量"] = True
    if x["recent_200_retest"]:
        flags["200MA 回踩再上"] = True
    if c > m5 > m10 > m20 > m60 > m200:
        flags["多頭排列"] = True
    if c > x["prev20high"] and x["volume_ratio"] >= 1.2:
        flags["20日突破+量能"] = True
    if x["bb_width"] <= 0.10 and c > x["bb_upper"]:
        flags["布林壓縮突破"] = True
    if x["above200"] and c > m20 and x["bias20"] <= 6:
        flags["強勢回調/回收"] = True
    return flags

def technical_score(x: dict) -> float:
    score=0.0
    score += 25 if x["above200"] else 0
    score += 20 if x["ma200_slope20"] > 0 else 0
    score += 15 if x["close"] > x["ma20"] else 0
    score += 15 if x["ma20"] > x["ma60"] > x["ma200"] else 0
    score += 10 if x["volume_ratio"] >= 1.2 else 5 if x["volume_ratio"] >= 1.0 else 0
    score += 10 if -0.02 <= x["bias20"]/100 <= 0.06 else 0
    score += 5 if x["close"] > x["prev20high"] else 0
    # Reward a fresh 200MA breakout, but keep the score capped at 100.
    if x["crossed_up_200"]:
        score += 10
    elif x["recent_200_breakout"]:
        score += 6
    return min(100.0, score)
