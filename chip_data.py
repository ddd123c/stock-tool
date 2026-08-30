# -*- coding: utf-8 -*-
"""神秘金字塔 >400 張大股東週資料抓取與特徵計算。"""
from __future__ import annotations
import re
from io import StringIO
from typing import Optional
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

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
    """抓神秘金字塔類股排行(週)全部股票，取最新一週 >400張大股東持有張數增減%。"""
    url = "https://norway.twsthr.info/StockHoldersTopWeek.aspx"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # 網站表格欄位是多層表頭，pd.read_html 在這個頁面容易因 rowspan/colspan
    # 產生不穩定的欄位結構，因此改從 HTML row 直接解析。
    m = re.search(r"收盤價日期\s*:\s*(\d{4}/\d{2}/\d{2})", soup.get_text(" ", strip=True))
    latest_date = pd.to_datetime(m.group(1), format="%Y/%m/%d") if m else pd.Timestamp.today().normalize()

    rows = []
    for tr in soup.find_all("tr"):
        link = None
        for a in tr.find_all("a", href=True):
            href = str(a.get("href"))
            if "StockHolders.aspx" in href and "STOCK=" in href.upper():
                link = a
                break
        if link is None:
            continue

        code = _stock_code(link.get_text(" ", strip=True))
        if not code:
            continue

        cells = tr.find_all(["td", "th"])
        link_cell_idx = next((i for i, td in enumerate(cells) if link in td.find_all("a", href=True)), None)
        if link_cell_idx is None:
            continue

        stock_name = re.sub(r"^\s*\d{4,6}", "", link.get_text(" ", strip=True)).strip()

        # 股票名稱後面依序是：類別、最近數週增減%、走勢、總增減、上週持有%、收盤價、漲跌。
        # 取類別後最前面的 6 個可解析百分比數字；最後一個就是最新一週。
        following = cells[link_cell_idx + 2:]
        weekly_values = []
        for td in following:
            txt = td.get_text(" ", strip=True).replace(",", "")
            if not txt or txt in {"-", "—", "–"}:
                continue
            mm = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*%?", txt)
            if mm:
                weekly_values.append(float(mm.group(1)))
                if len(weekly_values) == 6:
                    break

        if not weekly_values:
            continue

        rows.append({
            "代號": code,
            "股票": stock_name,
            "資料週": latest_date,
            "大股東週增減%": weekly_values[-1],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("找不到神秘金字塔週排行的股票資料；可能是網站表格格式更新。")

    return out.drop_duplicates("代號").reset_index(drop=True)

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
