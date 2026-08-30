# -*- coding: utf-8 -*-
"""神秘金字塔 >400 張大股東週資料抓取與特徵計算。"""
from __future__ import annotations
import re
from io import StringIO
from typing import Optional
import numpy as np
import pandas as pd
import requests

BASE_URL = "https://norway.twsthr.info/StockHolders.aspx"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36"}

def _stock_code(value: object) -> Optional[str]:
    m = re.search(r"(?<!\d)(\d{4,6})(?!\d)", str(value))
    return m.group(1) if m else None

def load_chip_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        s = str(c).strip()
        if "股票代號/名稱" in s: rename[c] = "stock"
        elif s == "類別": rename[c] = "industry"
        elif s == "總增減": rename[c] = "chip_total"
        elif "上週持有%" in s: rename[c] = "holder_pct"
        elif "今日收盤價" in s: rename[c] = "price"
        elif "今日漲跌" in s: rename[c] = "price_change"
    df = df.rename(columns=rename)
    if "stock" not in df.columns:
        raise ValueError("CSV 缺少 股票代號/名稱 欄位")
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

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join(str(v) for v in col if str(v).lower() != "nan").strip()
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out

def fetch_stock_weekly(code: str, weeks: int = 12) -> pd.DataFrame:
    """抓單一股票神秘金字塔個股頁的最近數週 >400張大股東持有百分比。"""
    code = str(code).strip()
    url = f"{BASE_URL}?STOCK={code}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))

    target = None
    for t in tables:
        t = _flatten_columns(t)
        joined = " ".join(t.columns)
        if "資料日期" in joined and "大股東" in joined and ("持有百分比" in joined or "持有百分比" in str(t.head(2).to_string())):
            target = t
            break
    if target is None:
        raise ValueError(f"{code}: 找不到大股東週資料表")

    # 找日期與 >400 張持有百分比欄位。
    date_col = next((c for c in target.columns if "資料日期" in str(c)), None)
    pct_col = next((c for c in target.columns if "大股東持有百分比" in str(c) and "1000" not in str(c)), None)
    if pct_col is None:
        # fallback: 最後一個名稱包含「持有百分比」且不是 >1000 張的欄位
        candidates = [c for c in target.columns if "持有百分比" in str(c) and "1000" not in str(c)]
        pct_col = candidates[0] if candidates else None
    if date_col is None or pct_col is None:
        raise ValueError(f"{code}: 找不到日期/大股東持有百分比欄位")

    out = pd.DataFrame({
        "date": target[date_col].astype(str).str.extract(r"(\d{8})", expand=False),
        "holder_pct": pd.to_numeric(target[pct_col], errors="coerce"),
    })
    out = out.dropna(subset=["date", "holder_pct"]).copy()
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date", ascending=False).head(weeks).sort_values("date")
    out["code"] = code
    return out[["code", "date", "holder_pct"]].reset_index(drop=True)

def fetch_chip_html(url: str = "https://norway.twsthr.info/StockHoldersTopWeek.aspx") -> pd.DataFrame:
    """保留舊功能：抓類股排行頁目前週資料。"""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    for t in tables:
        t = _flatten_columns(t)
        cols = " ".join(t.columns)
        if "股票代號" in cols and ("總增減" in cols or "持有率" in cols):
            t["code"] = t.iloc[:, 0].map(_stock_code)
            return normalize_chip_columns(t)
    raise ValueError("找不到大股東週排行表")


def fetch_all_weekly_chip_rankings() -> pd.DataFrame:
    """抓神秘金字塔週排行全部股票，取最新一週 >400張大股東持有張數增減率。"""
    url = "https://norway.twsthr.info/StockHoldersTopWeek.aspx"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    target = None
    for t in tables:
        t = _flatten_columns(t)
        if len(t) < 20:
            continue
        cols = " ".join(map(str, t.columns))
        if "股票代號" in cols and "大股東" in cols and "持有張數增減" in cols:
            target = t
            break
    if target is None:
        raise ValueError("找不到神秘金字塔週排行表")
    date_cols = []
    for c in target.columns:
        m = re.search(r"(20\d{6})", str(c))
        if m:
            date_cols.append((m.group(1), c))
    if not date_cols:
        raise ValueError("找不到週籌碼日期欄位")
    latest_date, latest_col = max(date_cols, key=lambda x: x[0])
    stock_col = target.columns[0]
    out = pd.DataFrame({
        "代號": target[stock_col].map(_stock_code),
        "股票": target[stock_col].astype(str).str.replace(r"^\s*\d{4,6}", "", regex=True).str.strip(),
        "大股東週增減%": pd.to_numeric(
            target[latest_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        ),
    })
    out["資料週"] = pd.to_datetime(latest_date, format="%Y%m%d")
    out = out.dropna(subset=["代號", "大股東週增減%"]).drop_duplicates("代號")
    return out[["代號", "股票", "資料週", "大股東週增減%"]].reset_index(drop=True)

def chip_features(weekly: pd.DataFrame) -> dict:
    """由單股週資料計算 1/4/8 週比例變化與趨勢。"""
    if weekly is None or weekly.empty:
        return {"chip_now": None, "chip_1w": None, "chip_4w": None, "chip_8w": None, "chip_trend": None}
    x = weekly.sort_values("date").dropna(subset=["holder_pct"])
    vals = x["holder_pct"].astype(float).tolist()
    if not vals:
        return {"chip_now": None, "chip_1w": None, "chip_4w": None, "chip_8w": None, "chip_trend": None}
    now = vals[-1]
    one = now - vals[-2] if len(vals) >= 2 else None
    four = now - vals[-5] if len(vals) >= 5 else None
    eight = now - vals[-9] if len(vals) >= 9 else None
    trend = float(np.polyfit(range(len(vals)), vals, 1)[0]) if len(vals) >= 2 else 0.0
    return {"chip_now": now, "chip_1w": one, "chip_4w": four, "chip_8w": eight, "chip_trend": trend}
