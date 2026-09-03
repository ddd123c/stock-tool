# -*- coding: utf-8 -*-
"""獨立量化研究模組。只讀取既有技術結果，不修改 200MA 核心算法。"""
from __future__ import annotations
import numpy as np
import pandas as pd

def research_features(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty:
        return None
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "close" not in x.columns or "volume" not in x.columns:
        return None
    close = pd.to_numeric(x["close"], errors="coerce").dropna()
    volume = pd.to_numeric(x["volume"], errors="coerce").reindex(close.index).fillna(0)
    if len(close) < 253:
        return None
    c = float(close.iloc[-1])
    ret20 = c / float(close.iloc[-21]) - 1
    ret60 = c / float(close.iloc[-61]) - 1
    ret120 = c / float(close.iloc[-121]) - 1
    ret252 = c / float(close.iloc[-253]) - 1
    ma20 = close.rolling(20).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    vol20 = volume.rolling(20).mean().iloc[-1]
    vol5 = volume.tail(5).mean()
    volume_shock = float(volume.iloc[-1] / vol20) if vol20 else 0.0
    volume_ratio_5_20 = float(vol5 / vol20) if vol20 else 0.0
    vol20_return = close.pct_change().tail(20).std() * np.sqrt(252)
    bias20 = (c / ma20 - 1) * 100 if ma20 else np.nan
    high20 = close.iloc[-21:-1].max()
    breakout20 = c > high20
    return {
        "ret20": float(ret20), "ret60": float(ret60), "ret120": float(ret120),
        "ret252": float(ret252), "volume_shock": volume_shock,
        "volume_ratio_5_20": volume_ratio_5_20, "volatility20_ann": float(vol20_return),
        "bias20": float(bias20), "breakout20": bool(breakout20),
        "above200": bool(c > ma200), "close": c, "ma20": float(ma20), "ma200": float(ma200),
    }

def score_features(f: dict) -> float:
    # 100分研究分數：只作排序，不代表預測勝率。
    score = 0.0
    score += 20 if f["breakout20"] else 0
    score += 10 if f["above200"] else 0
    score += 10 if f["ret20"] > 0 else 0
    score += 10 if f["ret60"] > 0 else 0
    score += 10 if f["ret120"] > 0 else 0
    score += 10 if f["ret252"] > 0 else 0
    score += 10 if f["volume_shock"] >= 2.0 else 5 if f["volume_shock"] >= 1.5 else 0
    score += 10 if 0 <= f["bias20"] <= 25 else 5 if -5 <= f["bias20"] < 0 else 0
    score += 10 if f["volatility20_ann"] < 0.60 else 5 if f["volatility20_ann"] < 0.90 else 0
    return min(100.0, score)

def research_summary(f: dict) -> str:
    return (
        f"20D {f['ret20']*100:+.1f}%｜60D {f['ret60']*100:+.1f}%｜"
        f"120D {f['ret120']*100:+.1f}%｜12M {f['ret252']*100:+.1f}%｜"
        f"單日量衝擊 {f['volume_shock']:.2f}x｜20D波動 {f['volatility20_ann']*100:.1f}%"
    )
