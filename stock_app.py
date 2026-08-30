import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import urllib3
from io import StringIO
from quant_engine import calculate_200ma_signal

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="台股 200MA 即時量化篩選 V2.8", layout="wide")


# 200MA 驗證：輸入基準股票後，可先確認「未還原、200交易日」計算。
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 200MA 基準驗證")
st.sidebar.caption("先用已知答案驗證計算，再進行全台股掃描。")

st.title("🚀 台股 Quant Screener V2.8")
st.caption("200MA 專用掃描：歷史日線 + 台灣交易所即時價格；法人／大股東資料完全不參與本頁掃描。")

st.sidebar.header("⚙️ 200MA 掃描設定")

source_option = st.sidebar.radio(
    "股票池",
    ("全台股", "手動輸入")
)

min_vol_limit = st.sidebar.number_input(
    "5日均量下限（張）",
    min_value=0,
    value=1000,
    step=500
)

st.sidebar.markdown("---")
st.sidebar.info(
    "規則：\n"
    "1. 現價必須站在 200MA 上方。\n"
    "2. 最近 5 個交易日內突破 200MA 的股票保留。\n"
    "3. 跌破 200MA 立即排除。\n"
    "4. 法人／大股東／新聞資料不在本頁抓取，避免拖慢 200MA。\n"
    "5. 不自動每 5 分鐘掃描，避免平台資源被吃光。"
)
