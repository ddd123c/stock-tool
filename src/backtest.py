from __future__ import annotations

import numpy as np
import pandas as pd


def convergence_events(df, z_entry=2.0, z_exit=0.5, max_holding_days=10):
    """Backtest positive-basis convergence using an already aligned daily dataset.

    Required columns: date, zscore, residual_basis.
    The strategy is research-only: it assumes a long-spot/short-futures spread
    and measures mark-to-market convergence of the residual basis, excluding
    commissions, financing, dividends and slippage unless supplied separately.
    """
    x = df.copy().sort_values("date").reset_index(drop=True)
    trades = []
    in_trade = False
    entry_i = None
    for i, row in x.iterrows():
        z = row.get("zscore", np.nan)
        if not in_trade and pd.notna(z) and z >= z_entry:
            in_trade = True
            entry_i = i
            continue
        if in_trade:
            held = i - entry_i
            exit_now = (pd.notna(z) and z <= z_exit) or held >= max_holding_days
            if exit_now:
                entry = x.loc[entry_i]
                exit_row = row
                rb0 = float(entry["residual_basis"])
                rb1 = float(exit_row["residual_basis"])
                pnl_proxy = rb0 - rb1
                trades.append({
                    "entry_date": entry["date"],
                    "exit_date": exit_row["date"],
                    "holding_days": held,
                    "entry_residual_basis": rb0,
                    "exit_residual_basis": rb1,
                    "basis_convergence_pnl_proxy": pnl_proxy,
                    "win": pnl_proxy > 0,
                })
                in_trade = False
                entry_i = None
    return pd.DataFrame(trades)


def summarize_trades(trades):
    if trades is None or trades.empty:
        return {"trades": 0, "win_rate": np.nan, "avg_pnl_proxy": np.nan, "total_pnl_proxy": 0.0}
    return {
        "trades": int(len(trades)),
        "win_rate": float(trades["win"].mean()),
        "avg_pnl_proxy": float(trades["basis_convergence_pnl_proxy"].mean()),
        "total_pnl_proxy": float(trades["basis_convergence_pnl_proxy"].sum()),
    }
