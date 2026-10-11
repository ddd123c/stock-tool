from __future__ import annotations

import numpy as np
import pandas as pd


def liquidity_ratio(volume, baseline_volume):
    v = pd.Series(volume, dtype=float)
    b = pd.Series(baseline_volume, index=v.index, dtype=float)
    return v / b.replace(0, np.nan)


def arbitrage_score(zscore, dte, liquidity_ratio_value, max_spread_pct=0.003):
    """Transparent 0-100 research score; not an execution recommendation."""
    z = float(zscore) if pd.notna(zscore) else np.nan
    d = float(dte) if pd.notna(dte) else np.nan
    liq = float(liquidity_ratio_value) if pd.notna(liquidity_ratio_value) else np.nan
    if np.isnan(z) or np.isnan(d) or np.isnan(liq):
        return np.nan
    score = 0.0
    if z >= 2.0:
        score += 25
    if z >= 2.5:
        score += 15
    if z >= 3.0:
        score += 10
    if d <= 10:
        score += 10
    if d <= 5:
        score += 10
    if liq >= 1.0:
        score += 15
    if liq >= 2.0:
        score += 5
    return float(min(score, 100.0))


def signal_label(zscore, score):
    if pd.isna(zscore) or pd.isna(score):
        return "INSUFFICIENT DATA"
    if zscore >= 2 and score >= 70:
        return "WATCH: HIGH POSITIVE BASIS"
    if zscore <= -2:
        return "WATCH: NEGATIVE BASIS"
    return "NEUTRAL"
