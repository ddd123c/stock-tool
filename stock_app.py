# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
from chip_data import fetch_chip_html, load_chip_csv, chip_features
from quant_engine import compute_indicators, technical_score, strategy_flags

st.set_page_config(page_title="台股 Quant Screener V2.1", layout="wide")
st.title("📊 台股 Quant Screener V2.1")
st.caption("200MA 突破/回踩 + 技術突破 + 大股東週籌碼 + 多因子評分")

with st.sidebar:
    st.header("⚙️ 掃描設定")
    universe = st.radio("股票池", ["全台股", "手動輸入"])
    min_vol = st.number_input("5日均量下限（張）", min_value=0, value=1000, step=500)
    chip_source = st.radio("籌碼來源", ["網站即時抓取", "CSV"])
    chip_csv = st.text_input("CSV 路徑", "神秘金字塔 - 股權類股排行.csv")
    w_tech = st.slider("技術面權重", 0.0, 1.0, 0.70, 0.05)
    w_chip = 1.0 - w_tech
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
def get_chip(source, path):
    return load_chip_csv(path) if source == "CSV" else fetch_chip_html()

stocks=get_stock_list()
if universe=="手動輸入":
    wanted={x.strip() for x in codes_text.split(",") if x.strip()}
    stocks=stocks[stocks["code"].isin(wanted)].copy()

stocks["ticker"]=stocks["code"].map(lambda x:f"{x}.TW")
tickers=stocks["ticker"].tolist()

if st.button("🚀 開始量化掃描", type="primary"):
    with st.spinner("下載股價、計算技術與籌碼因子..."):
        prices=get_prices(tuple(tickers))
        chip=get_chip(chip_source, chip_csv)
        rows=[]
        for _, stock in stocks.iterrows():
            code=str(stock["code"]); ticker=stock["ticker"]
            try:
                if len(tickers)==1: df=prices
                else:
                    if ticker not in prices.columns.levels[0]: continue
                    df=prices[ticker]
                x=compute_indicators(df)
                if not x or x["vol5"] < min_vol*1000: continue
                flags=strategy_flags(x)
                if not flags: continue

                if strategy_filter == "只看今日突破" and not x["crossed_up_200"]: continue
                if strategy_filter == "只看近5日突破" and not x["recent_200_breakout"]: continue
                if strategy_filter == "只看回踩再上" and not x["recent_200_retest"]: continue

                cr=chip[chip["code"].astype(str)==code]
                cf=chip_features(cr.iloc[0]) if not cr.empty else {}
                tech=technical_score(x)
                chip_score=None
                if cf.get("chip_4w") is not None:
                    chip_score=max(0,min(100,50+cf["chip_4w"]*3+cf["chip_trend"]*10))
                total=tech*w_tech+(chip_score if chip_score is not None else tech)*w_chip
                rows.append({
                    "代號":code,"股票":stock["name"],"Quant Score":round(total,1),
                    "技術分":round(tech,1),"籌碼分":None if chip_score is None else round(chip_score,1),
                    "收盤":round(x["close"],2),"200MA":round(x["ma200"],2),
                    "200MA狀態":x["breakout_type"],"站上200MA天數":x["above200_run"],
                    "200MA斜率20D%":round(x["ma200_slope20"]*100,2),"20MA乖離%":round(x["bias20"],2),
                    "量比(5日/20日)":round(x["volume_ratio"],2),"5日均量(張)":round(x["vol5"]/1000),
                    "籌碼1週":cf.get("chip_1w"),"籌碼2週":cf.get("chip_2w"),"籌碼4週":cf.get("chip_4w"),
                    "籌碼趨勢":cf.get("chip_trend"),"策略":", ".join(flags.keys())
                })
            except Exception:
                continue
    result=pd.DataFrame(rows)
    if result.empty:
        st.warning("目前沒有符合條件的標的。請先把「200MA策略」設為「全部」，確認資料正常。")
    else:
        result=result.sort_values(["Quant Score","技術分"],ascending=False)
        st.success(f"找到 {len(result)} 檔候選股")
        st.dataframe(result,use_container_width=True,hide_index=True)
        st.subheader("🏆 Top 20")
        st.dataframe(result.head(20),use_container_width=True,hide_index=True)
        st.info("Quant Score 是研究/篩選模型，不是保證獲利的買賣訊號。200MA突破目前定義為：前一交易日收盤在200MA以下/相等，當日收盤站上；另提供近5日突破與回踩再上分類。下一步再用歷史 snapshot 做 walk-forward 回測。")
