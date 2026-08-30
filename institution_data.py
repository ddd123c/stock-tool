# -*- coding: utf-8 -*-
"""台股三大法人每日買賣超：外資/投信/自營商連續買超天數。"""
from __future__ import annotations
from datetime import datetime, timedelta
import re
import requests
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36"}

def _num(v):
    return pd.to_numeric(str(v).replace(",", "").replace(" ", ""), errors="coerce")

def _code(v):
    m = re.search(r"(?<!\d)(\d{4,6})(?!\d)", str(v))
    return m.group(1) if m else None

def _parse_twse(date):
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    r = requests.get(url, params={"date": date.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    obj = r.json()
    fields = obj.get("fields", [])
    data = obj.get("data", [])
    if not fields or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=fields)
    code_col = next((c for c in fields if "證券代號" in str(c)), fields[0])
    foreign_col = next((c for c in fields if "外陸資買賣超股數" in str(c) and "不含" in str(c)), None)
    trust_col = next((c for c in fields if "投信買賣超股數" in str(c)), None)
    dealer_col = next((c for c in fields if "自營商買賣超股數" in str(c) and "自行買賣" not in str(c) and "避險" not in str(c)), None)
    if foreign_col is None or trust_col is None:
        return pd.DataFrame()
    dealer = df[dealer_col].map(_num) if dealer_col is not None else pd.Series(0, index=df.index)
    return pd.DataFrame({
        "代號": df[code_col].map(_code),
        "外資買賣超": df[foreign_col].map(_num),
        "投信買賣超": df[trust_col].map(_num),
        "自營商買賣超": dealer,
        "市場": "TWSE",
        "日期": date,
    }).dropna(subset=["代號"])

def _parse_tpex(date):
    # TPEx legacy JSON endpoint; d uses ROC date.
    roc = date.year - 1911
    d = f"{roc:03d}/{date.month:02d}/{date.day:02d}"
    url = "https://www.tpex.org.tw/web/stock/3insti/3insti_summary.php"
    r = requests.get(url, params={"l":"zh-tw","o":"json","d":d,"s":"0,asc","se":"EW","t":"D"}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    obj = r.json()
    tables = obj.get("tables", [])
    table = tables[0] if tables else obj
    fields = table.get("fields", []) if isinstance(table, dict) else []
    data = table.get("data", []) if isinstance(table, dict) else []
    if not data:
        data = obj.get("aaData", []) if isinstance(obj, dict) else []
    if not data:
        return pd.DataFrame()
    if fields:
        df = pd.DataFrame(data, columns=fields)
        code_col = next((c for c in fields if "代號" in str(c)), fields[0])
        foreign_col = next((c for c in fields if "外資" in str(c) and "買賣超" in str(c)), None)
        trust_col = next((c for c in fields if "投信" in str(c) and "買賣超" in str(c)), None)
        dealer_col = next((c for c in fields if "自營商" in str(c) and "買賣超" in str(c)), None)
    else:
        df = pd.DataFrame(data)
        code_col = df.columns[0]
        # TPEx columns follow code/name, foreign buy/sell/net, trust buy/sell/net.
        foreign_col = df.columns[4] if len(df.columns) > 4 else None
        trust_col = df.columns[7] if len(df.columns) > 7 else None
        dealer_col = None
        for i, c in enumerate(df.columns):
            if "自營商" in str(c) and "買賣超" in str(c):
                dealer_col = c
                break
    if foreign_col is None or trust_col is None:
        return pd.DataFrame()
    dealer = df[dealer_col].map(_num) if dealer_col is not None else pd.Series(0, index=df.index)
    return pd.DataFrame({
        "代號": df[code_col].map(_code),
        "外資買賣超": df[foreign_col].map(_num),
        "投信買賣超": df[trust_col].map(_num),
        "自營商買賣超": dealer,
        "市場": "TPEx",
        "日期": date,
    }).dropna(subset=["代號"])

def _daily(date):
    frames = []
    for fn in (_parse_twse, _parse_tpex):
        try:
            x = fn(date)
            if not x.empty:
                frames.append(x)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def get_institution_streaks(codes, lookback_days=45):
    """回傳指定股票截至最新可取得交易日的外資/投信/自營商連買天數與累積買超。"""
    wanted = {str(x).strip() for x in codes}
    end = datetime.now().date()
    rows = []
    frames = []
    for i in range(lookback_days):
        d = end - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        x = _daily(d)
        if not x.empty:
            frames.append(x[x["代號"].isin(wanted)].copy())
        # 只要已經取得足夠的連續交易資料即可提前結束；保留 30 個日曆日以上的彈性。
        if len(frames) >= 25:
            break
    if not frames:
        return pd.DataFrame({"代號": list(wanted), "法人連買天數": 0, "外資連買天數": 0, "投信連買天數": 0, "自營商連買天數": 0,
                             "外資5日累計(張)": 0, "投信5日累計(張)": 0, "自營商5日累計(張)": 0})
    allx = pd.concat(frames, ignore_index=True)
    allx["日期"] = pd.to_datetime(allx["日期"])
    allx = allx.sort_values(["代號","日期"], ascending=[True,False])

    out = []
    for code, g in allx.groupby("代號"):
        g = g.drop_duplicates("日期").sort_values("日期", ascending=False)
        foreign = g["外資買賣超"].fillna(0).tolist()
        trust = g["投信買賣超"].fillna(0).tolist()
        dealer = g["自營商買賣超"].fillna(0).tolist()
        total = [f + t + d for f, t, d in zip(foreign, trust, dealer)]
        total_run = 0
        f_run = 0
        t_run = 0
        d_run = 0
        for v in total:
            if v > 0: total_run += 1
            else: break
        for v in foreign:
            if v > 0: f_run += 1
            else: break
        for v in trust:
            if v > 0: t_run += 1
            else: break
        for v in dealer:
            if v > 0: d_run += 1
            else: break
        out.append({
            "代號": code,
            "法人連買天數": total_run,
            "外資連買天數": f_run,
            "投信連買天數": t_run,
            "自營商連買天數": d_run,
            "外資5日累計(張)": round(sum(foreign[:5]) / 1000, 1),
            "投信5日累計(張)": round(sum(trust[:5]) / 1000, 1),
            "自營商5日累計(張)": round(sum(dealer[:5]) / 1000, 1),
        })
    return pd.DataFrame(out)
