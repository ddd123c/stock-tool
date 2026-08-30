# -*- coding: utf-8 -*-
"""Technical indicators and scoring for the Quant Screener V2."""
from __future__ import annotations
import numpy as np
import pandas as pd

MA_WINDOWS = (5, 10, 15, 20, 60, 200)

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def compute_indicators(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty:
        return None
    x = df.copy()
    # yfinance can return duplicate/odd column labels; normalize to OHLCV.
    x.columns = [str(c).strip().lower() for c in x.columns]
    for c in ("close", "volume"):
        if c not in x.columns:
            return None
        x[c] = _num(x[c])
    close = x["close"].dropna()
    if len(close) < 200:
        return None
    volume = _num(x["volume"]).fillna(0)
    ma = {n: close.rolling(n).mean().iloc[-1] for n in MA_WINDOWS}
    ma200_series = close.rolling(200).mean()
    if pd.isna(ma[200]):
        return None
    slope_ref = ma200_series.iloc[-21]
    slope20 = (ma[200] / slope_ref - 1.0) if pd.notna(slope_ref) and slope_ref else 0.0
    ma20 = ma[20]
    std20 = close.rolling(20).std().iloc[-1]
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    width = (upper - lower) / ma20 if ma20 else np.nan
    vol5 = volume.tail(5).mean()
    vol20 = volume.tail(20).mean()
    prev20high = close.iloc[-21:-1].max() if len(close) >= 21 else np.nan
    return {
        "close": float(close.iloc[-1]), "ma5": float(ma[5]), "ma10": float(ma[10]),
        "ma15": float(ma[15]), "ma20": float(ma[20]), "ma60": float(ma[60]),
        "ma200": float(ma[200]), "ma200_slope20": float(slope20),
        "bias20": float((close.iloc[-1] / ma20 - 1) * 100),
        "vol5": float(vol5), "vol20": float(vol20),
        "volume_ratio": float(vol5 / vol20) if vol20 else 0.0,
        "bb_upper": float(upper), "bb_lower": float(lower),
        "bb_width": float(width) if pd.notna(width) else 0.0,
        "prev20high": float(prev20high),
        "above200": bool(close.iloc[-1] > ma[200]),
    }

def strategy_flags(x: dict) -> dict:
    c=x["close"]; m5=x["ma5"]; m10=x["ma10"]; m15=x["ma15"]; m20=x["ma20"]; m60=x["ma60"]; m200=x["ma200"]
    flags={}
    if x["above200"] and x["ma200_slope20"] > 0:
        flags["200MA 趨勢"] = True
    if c > m5 > m10 > m20 > m60 > m200:
        flags["多頭排列"] = True
    if c > x["prev20high"] and x["volume_ratio"] >= 1.2:
        flags["20日突破+量能"] = True
    if x["bb_width"] <= 0.10 and c > x["bb_upper"]:
        flags["布林壓縮突破"] = True
    # Pullback recovery: price is back above 20MA while still above 200MA.
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
    return min(100.0, score)
