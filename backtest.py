# -*- coding: utf-8 -*-
"""Historical backtesting engine for quant-research-v1.

Design goals:
- Use only information available at the signal close (no look-ahead).
- Enter at the next trading day's Open.
- Exit at the Close after N completed holding days.
- Use raw OHLCV data; adjusted close is not used for execution prices.
- Each ticker is tested independently; overlapping signals are allowed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        # A single-ticker yfinance download may still return a MultiIndex.
        if len(x.columns.get_level_values(0).unique()) == 1:
            x.columns = x.columns.get_level_values(-1)
        else:
            raise ValueError("backtest expects one ticker at a time")
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(x.columns):
        return pd.DataFrame()
    for c in required:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["open", "close"]).copy()
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x


def _signal_frame(
    df: pd.DataFrame,
    breakout_lookback: int = 20,
    volume_multiple: float = 2.5,
    require_above200: bool = True,
    max_bias20: float | None = 25.0,
) -> pd.DataFrame:
    """Build signals using data through each row only."""
    x = _clean_ohlcv(df)
    if x.empty:
        return x

    x["ma20"] = x["close"].rolling(20, min_periods=20).mean()
    x["ma200"] = x["close"].rolling(200, min_periods=200).mean()
    x["vol20"] = x["volume"].rolling(20, min_periods=20).mean()
    x["prev_high20"] = x["close"].shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).max()
    x["bias20"] = (x["close"] / x["ma20"] - 1.0) * 100.0

    # All conditions are evaluated at day t close. No future row is used.
    x["breakout20"] = x["close"] > x["prev_high20"]
    x["volume_ok"] = x["volume"] > x["vol20"] * float(volume_multiple)
    x["above200"] = x["close"] > x["ma200"]
    x["bias_ok"] = True if max_bias20 is None else x["bias20"] <= float(max_bias20)

    x["signal"] = x["breakout20"] & x["volume_ok"]
    if require_above200:
        x["signal"] &= x["above200"]
    x["signal"] &= x["bias_ok"]
    return x


def backtest_one(
    df: pd.DataFrame,
    ticker: str = "",
    holding_days: int = 5,
    breakout_lookback: int = 20,
    volume_multiple: float = 2.5,
    require_above200: bool = True,
    max_bias20: float | None = 25.0,
) -> pd.DataFrame:
    """Return one row per completed trade.

    Signal is known after the signal day's close.
    Entry is the next trading day's Open.
    Exit is the Close of the Nth completed holding day.
    Incomplete trades at the end of the dataset are excluded from performance stats.
    """
    if holding_days < 1:
        raise ValueError("holding_days must be >= 1")
    x = _signal_frame(
        df,
        breakout_lookback=breakout_lookback,
        volume_multiple=volume_multiple,
        require_above200=require_above200,
        max_bias20=max_bias20,
    )
    if x.empty:
        return pd.DataFrame()

    rows = []
    idx = list(x.index)
    for i, signal_date in enumerate(idx):
        if not bool(x.at[signal_date, "signal"]):
            continue
        entry_i = i + 1
        exit_i = entry_i + holding_days - 1
        if exit_i >= len(idx):
            # Cannot know the completed holding-period return yet.
            continue
        entry_date = idx[entry_i]
        exit_date = idx[exit_i]
        entry = float(x.iloc[entry_i]["open"])
        exit_price = float(x.iloc[exit_i]["close"])
        if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0:
            continue
        ret = exit_price / entry - 1.0
        rows.append(
            {
                "ticker": ticker,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry,
                "exit_price": exit_price,
                "return": ret,
                "win": ret > 0,
                "breakout20": bool(x.iloc[i]["breakout20"]),
                "volume_multiple": float(x.iloc[i]["volume"] / x.iloc[i]["vol20"]) if x.iloc[i]["vol20"] else np.nan,
                "bias20": float(x.iloc[i]["bias20"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame, periods_per_year: int = 252) -> dict:
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "best_return": np.nan,
            "worst_return": np.nan,
            "return_std": np.nan,
            "profit_factor": np.nan,
            "sharpe": np.nan,
            "compound_return": np.nan,
        }
    r = pd.to_numeric(trades["return"], errors="coerce").dropna()
    if r.empty:
        return {"trades": 0}
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    profit_factor = gains / losses if losses > 0 else np.inf
    # Trade-level Sharpe is only a descriptive statistic, not a portfolio Sharpe.
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year / max(1, len(r))) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "best_return": float(r.max()),
        "worst_return": float(r.min()),
        "return_std": float(r.std(ddof=1)) if len(r) > 1 else np.nan,
        "profit_factor": float(profit_factor),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "compound_return": float((1.0 + r).prod() - 1.0),
    }


def benchmark_return(df: pd.DataFrame) -> float | None:
    x = _clean_ohlcv(df)
    if len(x) < 2:
        return None
    first = float(x.iloc[0]["close"])
    last = float(x.iloc[-1]["close"])
    if first <= 0:
        return None
    return last / first - 1.0
