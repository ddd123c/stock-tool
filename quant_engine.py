# -*- coding: utf-8 -*-
"""Technical indicators, 200MA breakout states and scoring for Quant Screener V2."""
from __future__ import annotations
import numpy as np
import pandas as pd

MA_WINDOWS = (5, 10, 15, 20, 60, 200)

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def compute_indicators(df: pd.DataFrame, live_price: float | None = None) -> dict | None:
    if df is None or df.empty:
        return None
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    for c in ("close", "volume"):
        if c not in x.columns:
            return None
        x[c] = _num(x[c])

    # Yahoo Finance 的 Close 已反映股票分割後的歷史價格尺度。
    # 這裡刻意使用 Close，不再依 actions 的 stock splits 二次除權，
    # 否則像 0052 這類曾分割的標的會被重複調整，200MA 會明顯偏低。
    # 同時不使用 Adj Close，因此不把股利再還原進技術均線。
    x["close_raw"] = _num(x["close"])
    x = x.dropna(subset=["close_raw"])
    close = x["close_raw"]
    volume = x["volume"].fillna(0)

    if len(close) < 199:
        return None

    # 盤中：今天視為新的第 1 個交易日資料點。
    # 因此即時 200MA = 前 199 個已完成交易日 Close + 今天最新成交價。
    # 下一個開盤日開始，今天就自然成為歷史資料中的第 201 個點，
    # 滾動視窗會自動扣掉最舊的一天。
    # 不使用 Adj Close，也不再對 stock split 做第二次調整。
    use_live = live_price is not None
    historical_close = close
    if use_live and len(close) >= 1:
        # Yahoo daily history may or may not already contain today's partial
        # row. Only remove it when its date is actually today in Taiwan.
        last_ts = pd.Timestamp(x.index[-1])
        if last_ts.tzinfo is not None:
            last_date = last_ts.tz_convert("Asia/Taipei").date()
        else:
            last_date = last_ts.date()
        today_tw = pd.Timestamp.now(tz="Asia/Taipei").date()
        if last_date == today_tw:
            historical_close = close.iloc[:-1]

    if use_live and len(historical_close) < 199:
        return None

    if use_live:
        # Keep the full historical series so prior 200MA values remain
        # available for 5-day breakout detection and 20-day slope.
        # Today's live price replaces today's partial daily close (if present).
        analysis_close = pd.concat(
            [historical_close.reset_index(drop=True),
             pd.Series([float(live_price)])],
            ignore_index=True
        )
    else:
        analysis_close = close.reset_index(drop=True)

    if len(analysis_close) < 200:
        return None

    ma = {n: analysis_close.rolling(n).mean() for n in MA_WINDOWS}
    m = {n: float(ma[n].iloc[-1]) for n in MA_WINDOWS}
    ma200 = ma[200]
    ma200_now = m[200]
    ma200_5d = float(ma200.iloc[-6]) if len(ma200) >= 206 and pd.notna(ma200.iloc[-6]) else np.nan
    ma200_20d = float(ma200.iloc[-21]) if len(ma200) >= 221 and pd.notna(ma200.iloc[-21]) else np.nan

    slope20 = (ma200_now / ma200_20d - 1.0) if pd.notna(ma200_20d) and ma200_20d else 0.0
    slope5 = (ma200_now / ma200_5d - 1.0) if pd.notna(ma200_5d) and ma200_5d else 0.0

    c = float(analysis_close.iloc[-1])
    prev_c = float(analysis_close.iloc[-2]) if len(analysis_close) >= 2 else np.nan
    prev_ma200 = float(ma200.iloc[-2]) if len(ma200) >= 2 and pd.notna(ma200.iloc[-2]) else np.nan

    crossed_up = pd.notna(prev_c) and pd.notna(prev_ma200) and prev_c <= prev_ma200 and c > ma200_now
    crossed_down = pd.notna(prev_c) and pd.notna(prev_ma200) and prev_c >= prev_ma200 and c < ma200_now

    above = analysis_close > ma200
    run = 0
    current_above = bool(above.iloc[-1])
    for v in reversed(above.tolist()):
        if bool(v) == current_above:
            run += 1
        else:
            break

    vol5 = float(volume.tail(5).mean())
    vol20 = float(volume.tail(20).mean())
    volume_ratio = vol5 / vol20 if vol20 else 0.0

    ma20 = m[20]
    std20 = float(analysis_close.rolling(20).std().iloc[-1])
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    width = (upper - lower) / ma20 if ma20 else np.nan
    prev20high = float(analysis_close.iloc[-21:-1].max()) if len(analysis_close) >= 21 else np.nan

    recent_cross = False
    cross_days_ago = None
    max_lookback = min(5, len(analysis_close) - 1)
    for i in range(1, max_lookback + 1):
        if analysis_close.iloc[-i-1] <= ma200.iloc[-i-1] and analysis_close.iloc[-i] > ma200.iloc[-i]:
            recent_cross = True
            cross_days_ago = i - 1
            break

    recent_retest = False
    if c > ma200_now and len(analysis_close) >= 6:
        for i in range(1, min(6, len(analysis_close)-1)):
            if analysis_close.iloc[-i] <= ma200.iloc[-i] * 1.01:
                recent_retest = True
                break

    if crossed_up:
        breakout_type = "今日突破200MA"
    elif recent_cross:
        breakout_type = f"近5日突破200MA（{cross_days_ago}日前）"
    elif recent_retest and run >= 1:
        breakout_type = "200MA回踩後站回"
    elif current_above:
        breakout_type = "站上200MA（非新突破）"
    else:
        breakout_type = "200MA下方"

    return {
        "close": c, "prev_close": prev_c,
        "ma5": m[5], "ma10": m[10], "ma15": m[15], "ma20": m[20], "ma60": m[60], "ma200": ma200_now,
        "ma200_slope5": float(slope5), "ma200_slope20": float(slope20),
        "bias20": float((c / ma20 - 1) * 100),
        "vol5": vol5, "vol20": vol20, "volume_ratio": float(volume_ratio),
        "bb_upper": float(upper), "bb_lower": float(lower),
        "bb_width": float(width) if pd.notna(width) else 0.0,
        "prev20high": prev20high, "above200": current_above,
        "above200_run": run, "crossed_up_200": bool(crossed_up), "crossed_down_200": bool(crossed_down),
        "recent_200_breakout": bool(recent_cross), "cross_days_ago": cross_days_ago,
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
