from __future__ import annotations

import numpy as np
import pandas as pd


def theoretical_futures_price(spot, annual_rate, dividend_yield, dte):
    """Continuous cost-of-carry fair value. dte is calendar days."""
    spot = pd.Series(spot, dtype=float)
    t = pd.Series(dte, index=spot.index, dtype=float) / 365.0
    return spot * np.exp((annual_rate - dividend_yield) * t)


def basis(futures, spot):
    return pd.Series(futures, dtype=float) / pd.Series(spot, dtype=float) - 1.0


def residual_basis(futures, spot, annual_rate, dividend_yield, dte):
    fair = theoretical_futures_price(spot, annual_rate, dividend_yield, dte)
    return pd.Series(futures, dtype=float) / fair - 1.0


def rolling_zscore(series, window=60, min_periods=None):
    s = pd.Series(series, dtype=float)
    if min_periods is None:
        min_periods = max(20, window // 3)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=1)
    return (s - mean) / std.replace(0, np.nan)


def annualized_basis(futures, spot, dte):
    raw = basis(futures, spot)
    d = pd.Series(dte, index=raw.index, dtype=float).clip(lower=1)
    return raw * 365.0 / d


def convergence_fraction(basis_now, basis_later):
    """Positive value means the absolute basis moved toward zero."""
    now = pd.Series(basis_now, dtype=float)
    later = pd.Series(basis_later, index=now.index, dtype=float)
    denom = now.abs().replace(0, np.nan)
    return ((now.abs() - later.abs()) / denom).clip(lower=-10, upper=10)
