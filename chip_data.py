# -*- coding: utf-8 -*-
"""Weekly major-holder (大股東) data ingestion."""
from __future__ import annotations
import re
from io import StringIO
from typing import Optional
import pandas as pd
import requests
SOURCE_URL = "https://norway.twsthr.info/StockHoldersTopWeek.aspx"

def _stock_code(value: object) -> Optional[str]:
    m = re.match(r"\s*(\d{4})", str(value))
    return m.group(1) if m else None

def load_chip_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        s = str(c).strip()
        if "股票代號/名稱" in s: rename[c] = "stock"
        elif s == "類別": rename[c] = "industry"
        elif s == "總增減": rename[c] = "chip_total"
        elif s == "上週持有%": rename[c] = "holder_pct"
        elif s == "今日收盤價": rename[c] = "price"
        elif s == "今日漲跌": rename[c] = "price_change"
    df = df.rename(columns=rename)
    if "stock" not in df: raise ValueError("CSV 缺少 股票代號/名稱 欄位")
    df["code"] = df["stock"].map(_stock_code)
    return normalize_chip_columns(df)

def normalize_chip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    weekly = []
    for c in out.columns:
        if re.fullmatch(r"\d{4,8}", str(c).strip()):
            weekly.append(c)
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out.attrs["weekly_columns"] = weekly
    return out

def fetch_chip_html(url: str = SOURCE_URL) -> pd.DataFrame:
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("股票代號" in c for c in cols) and any("總增減" in c for c in cols):
            t = t.copy()
            t.columns = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
            return normalize_chip_columns(t)
    raise ValueError("找不到大股東週資料表")

def chip_features(row: pd.Series) -> dict:
    vals = []
    for c in row.index:
        if re.fullmatch(r"\d{4,8}", str(c).strip()):
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v): vals.append(float(v))
    if not vals:
        return {"chip_1w":None,"chip_2w":None,"chip_4w":None,"chip_trend":None}
    s = pd.Series(vals)
    import numpy as np
    trend = float(np.polyfit(range(len(s)), s.values, 1)[0]) if len(s) >= 2 else 0.0
    return {"chip_1w":vals[-1],"chip_2w":float(s.tail(2).sum()),
            "chip_4w":float(s.tail(4).sum()),"chip_trend":trend}
