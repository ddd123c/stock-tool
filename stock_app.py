# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from chip_data import fetch_stock_weekly, chip_features
from quant_engine import compute_indicators, technical_score, strategy_flags

st.set_page_config(page_title="台股 Quant Screener V2.2", layout="wide")
st.title("📊 台股 Quant Screener V2.2")
st.caption("200MA 突破/回踩 + 技術面 + 神秘金字塔 >400張大股東週籌碼")

with st.sidebar:
    st.header("⚙️ 掃描設定")
    universe = st.radio("股票池", ["全台股", "手動輸入"])
    min_vol = st.number_input("5日均量下限（張）", min_value=0, value=1000, step=500)
    w_tech = st.slider("技術面權重", 0.0, 1.0, 0.70, 0.05)
    strategy_filter = st.selectbox("200MA策略", ["全部", "只看今日突破", "只看近5日突破", "只看回踩再上"])
    codes_text = st.text_area("手動代號", "2330,2603,3017,3605,6446")

@st.cache_data(ttl=3600)
def get_stock_list():
    return pd.read_csv("tw_stocks.csv", dtype={"code": str})

@st.cache_data(ttl=1800)
def get_prices(tickers):
    return yf.download(list(tickers), period="2y", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)

@st.cache_data(ttl=1800)
def get_chip_weekly(code, weeks=12):
    return fetch_stock_weekly(code, weeks)

def scan_technical(stocks, min_vol, strategy_filter):
    tickers = stocks["ticker"].tolist()
    prices = get_prices(tuple(tickers))
    rows = []
    for _, stock in stocks.iterrows():
        code = str(stock["code"])
        ticker = stock["ticker"]
        try:
            if len(tickers) == 1:
                df = prices
            else:
                if not hasattr(prices, "columns") or ticker not in prices.columns.levels[0]:
                    continue
                df = prices[ticker]
            x = compute_indicators(df)
            if not x or x["vol5"] < min_vol * 1000:
                continue
            flags = strategy_flags(x)
            if not flags:
                continue
            if strategy_filter == "只看今日突破" and not x["crossed_up_200"]:
                continue
            if strategy_filter == "只看近5日突破" and not x["recent_200_breakout"]:
                continue
            if strategy_filter == "只看回踩再上" and not x["recent_200_retest"]:
                continue
            rows.append({
                "代號": code, "股票": stock["name"],
                "技術分": round(technical_score(x), 1),
                "收盤": round(x["close"], 2), "200MA": round(x["ma200"], 2),
                "200MA狀態": x["breakout_type"], "站上200MA天數": x["above200_run"],
                "200MA斜率20D%": round(x["ma200_slope20"] * 100, 2),
                "量比(5日/20日)": round(x["volume_ratio"], 2),
                "5日均量(張)": round(x["vol5"] / 1000),
                "策略": ", ".join(flags.keys())
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def add_chip_to_result(result, w_tech):
    rows = []
    for _, r in result.iterrows():
        try:
            weekly = get_chip_weekly(str(r["代號"]), 12)
            cf = chip_features(weekly)
        except Exception:
            cf = {}
        chip_score = None
        if cf.get("chip_now") is not None:
            # 以 1/4/8 週持有比例變化衡量籌碼方向；不把絕對高持股率誤當成「正在增加」。
            d1 = cf.get("chip_1w") or 0
            d4 = cf.get("chip_4w") or 0
            trend = cf.get("chip_trend") or 0
            chip_score = max(0, min(100, 50 + d1 * 12 + d4 * 4 + trend * 20))
        tech = float(r["技術分"])
        total = tech * w_tech + (chip_score if chip_score is not None else tech) * (1 - w_tech)
        item = r.to_dict()
        item.update({
            "Quant Score": round(total, 1),
            "大股東本週%": None if cf.get("chip_now") is None else round(cf["chip_now"], 2),
            "大股東1週變化%": None if cf.get("chip_1w") is None else round(cf["chip_1w"], 2),
            "大股東4週變化%": None if cf.get("chip_4w") is None else round(cf["chip_4w"], 2),
            "大股東8週變化%": None if cf.get("chip_8w") is None else round(cf["chip_8w"], 2),
            "籌碼趨勢": None if cf.get("chip_trend") is None else round(cf["chip_trend"], 3),
            "籌碼分": None if chip_score is None else round(chip_score, 1),
        })
        rows.append(item)
    return pd.DataFrame(rows)

stocks = get_stock_list()
if universe == "手動輸入":
    wanted = {x.strip() for x in codes_text.split(",") if x.strip()}
    stocks = stocks[stocks["code"].isin(wanted)].copy()
stocks["ticker"] = stocks["code"].map(lambda x: f"{x}.TW")

tab1, tab2 = st.tabs(["🚀 200MA量化篩選", "📈 大股東週籌碼"])

with tab1:
    if st.button("🚀 開始量化掃描", type="primary"):
        with st.spinner("先掃描技術面，再查符合條件股票的神秘金字塔週籌碼..."):
            result = scan_technical(stocks, min_vol, strategy_filter)
            if result.empty:
                st.warning("目前沒有符合條件的標的。可先把「200MA策略」設為「全部」，確認資料正常。")
            else:
                result = add_chip_to_result(result, w_tech)
                result = result.sort_values(["Quant Score", "技術分"], ascending=False)
                st.success(f"技術面找到 {len(result)} 檔，並逐檔補上神秘金字塔週籌碼")
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.subheader("🏆 Top 20")
                st.dataframe(result.head(20), use_container_width=True, hide_index=True)
                st.info("神秘金字塔籌碼欄位是 >400 張大股東持有比例；1/4/8 週變化都是「持有比例百分點的變化」，不是報酬率。Quant Score 僅供研究篩選。")

with tab2:
    code = st.text_input("輸入股票代號", "2330")
    weeks = st.slider("顯示最近幾週", 4, 12, 12)
    if st.button("📈 查詢大股東週籌碼"):
        try:
            weekly = get_chip_weekly(code, weeks)
            cf = chip_features(weekly)
            st.success(f"{code} 共取得 {len(weekly)} 週資料")
            st.dataframe(weekly.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
            if not weekly.empty:
                chart = weekly.sort_values("date")
                fig = px.line(chart, x="date", y="holder_pct", markers=True,
                              title=f"{code} >400張大股東持有比例")
                fig.update_yaxes(title="持有比例 (%)")
                fig.update_xaxes(title="日期")
                st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("本週持有比例", f"{cf['chip_now']:.2f}%" if cf["chip_now"] is not None else "—")
            c2.metric("1週變化", f"{cf['chip_1w']:+.2f}%" if cf["chip_1w"] is not None else "—")
            c3.metric("4週變化", f"{cf['chip_4w']:+.2f}%" if cf["chip_4w"] is not None else "—")
            c4.metric("8週變化", f"{cf['chip_8w']:+.2f}%" if cf["chip_8w"] is not None else "—")
        except Exception as e:
            st.error(f"抓取失敗：{e}")

st.caption("資料來源：神秘金字塔股權分散表；技術資料：Yahoo Finance。")
