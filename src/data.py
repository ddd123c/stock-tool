from __future__ import annotations

import pandas as pd


def normalize_ohlcv(df):
    """Normalize common OHLCV column names without making assumptions about source."""
    x = df.copy()
    x.columns = [str(c).strip().lower().replace(" ", "_") for c in x.columns]
    aliases = {
        "datetime": "date", "timestamp": "date", "time": "date",
        "close_price": "close", "last": "close", "last_price": "close",
        "vol": "volume", "qty": "volume",
    }
    x = x.rename(columns={c: aliases.get(c, c) for c in x.columns})
    if "date" in x:
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.sort_values("date") if "date" in x else x


def align_spot_futures(spot, futures, on="date"):
    s = normalize_ohlcv(spot).rename(columns={"close": "spot_close", "volume": "spot_volume"})
    f = normalize_ohlcv(futures).rename(columns={"close": "futures_close", "volume": "futures_volume"})
    return pd.merge_asof(
        s.sort_values(on), f.sort_values(on), on=on, direction="backward"
    )
