# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from chip_data import fetch_stock_weekly, chip_features, fetch_all_weekly_chip_rankings
from quant_engine import compute_indicators, technical_score, strategy_flags

st.set_page_config(page_title="台股 Quant Screener V2.5", layout="wide")
st.title("📊 台股 Quant Screener V2.4")
st.caption("200MA 即時技術篩選 ＋ 神秘金字塔每週大股東籌碼（兩個獨立功能）")

with st.sidebar:
    st.header("⚙️ 200MA 掃描設定")
    universe = st.radio("股票池", ["全台股", "手動輸入"])
    min_vol = st.number_input("5日均量下限（張）", min_value=0, value=1000, step=500)
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
            # 只保留剛站上 200MA 的標的：站上時間不得超過 5 個交易日。
            if x["above200_run"] > 5:
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

stocks = get_stock_list()
if universe == "手動輸入":
    wanted = {x.strip() for x in codes_text.split(",") if x.strip()}
    stocks = stocks[stocks["code"].isin(wanted)].copy()
stocks["ticker"] = stocks["code"].map(lambda x: f"{x}.TW")

tab1, tab2 = st.tabs(["🚀 200MA 即時量化篩選", "📈 神秘金字塔｜每週大股東"])

with tab1:
    st.info("這一頁專注技術面：只保留站上 200MA 不超過 5 個交易日的標的，適合盤中/盤後快速查看。這裡不混入每週大股東資料。")
    if st.button("🚀 開始量化掃描", type="primary"):
        with st.spinner("正在掃描台股技術面資料..."):
            result = scan_technical(stocks, min_vol, strategy_filter)
            if result.empty:
                st.warning("目前沒有符合條件的標的。可先把「200MA策略」設為「全部」，確認資料正常。")
            else:
                result = result.sort_values(["技術分"], ascending=False)
                st.success(f"找到 {len(result)} 檔符合 200MA 技術條件的標的")
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.subheader("🏆 Top 20")
                st.dataframe(result.head(20), use_container_width=True, hide_index=True)
                st.caption("技術資料：Yahoo Finance。此頁不等待神秘金字塔週資料，因此兩個功能彼此獨立。")


with tab2:
    st.info("這一頁專門看神秘金字塔每週籌碼：一次抓全部股票，再分成「增加最多 Top 20」與「減少最多 Top 20」。不與 200MA 即時篩選混在一起。")
    if st.button("📈 更新本週大股東排行", type="primary"):
        try:
            with st.spinner("正在抓取神秘金字塔全部股票的最新一週籌碼..."):
                chip = fetch_all_weekly_chip_rankings()
            latest = chip["資料週"].iloc[0].strftime("%Y/%m/%d")
            st.success(f"共取得 {len(chip)} 檔股票｜最新資料週：{latest}")
            inc = chip.sort_values("大股東週增減%", ascending=False).head(20)
            dec = chip.sort_values("大股東週增減%", ascending=True).head(20)
            left, right = st.columns(2)
            with left:
                st.subheader("🟢 大股東增加最多 Top 20")
                st.dataframe(inc, use_container_width=True, hide_index=True)
            with right:
                st.subheader("🔴 大股東減少最多 Top 20")
                st.dataframe(dec, use_container_width=True, hide_index=True)
            st.caption("排名依神秘金字塔最新一週「>400張大股東持有張數增減率」排序；增加與減少各取 20 檔。")
        except Exception as e:
            st.error(f"抓取失敗：{e}")

st.caption("台股代號來源：tw_stocks.csv｜技術資料：Yahoo Finance｜週籌碼：神秘金字塔")
