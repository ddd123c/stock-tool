import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="台股股期套利 Scanner", page_icon="📈", layout="wide")

st.title("📈 台股個股期貨套利 Scanner")
st.caption("Research prototype：現貨／股期基差、Residual Basis、Z-score 與到期收斂分析")

st.info("目前先提供可運行的研究框架。尚未接入即時行情 API，避免把示範資料誤當成實盤訊號。")

with st.sidebar:
    st.header("策略參數")
    z_threshold = st.slider("Z-score 門檻", 0.5, 4.0, 2.0, 0.1)
    max_dte = st.slider("最大剩餘到期日", 1, 60, 10)
    window = st.slider("Z-score 回看視窗", 20, 120, 60, 5)
    st.divider()
    st.caption("策略概念：Z > 門檻 且 DTE 較低時，研究『買現貨＋空股期』的基差收斂機會。")


def demo_data(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    spot = 1000 + np.cumsum(rng.normal(0, 8, n))
    basis = 0.004 + rng.normal(0, 0.0025, n)
    futures = spot * (1 + basis)
    return pd.DataFrame({"date": dates, "spot": spot, "futures": futures, "basis": basis})


df = demo_data()
df["rolling_mean"] = df["basis"].rolling(window).mean()
df["rolling_std"] = df["basis"].rolling(window).std(ddof=0)
df["zscore"] = (df["basis"] - df["rolling_mean"]) / df["rolling_std"]

# Demonstration DTE: for research only, based on nearest 3rd Wednesday.
def nearest_expiry(date: pd.Timestamp) -> pd.Timestamp:
    month_start = pd.Timestamp(date.year, date.month, 1)
    first = month_start
    wednesdays = pd.date_range(first, first + pd.offsets.MonthEnd(0), freq="W-WED")
    expiry = wednesdays[2]
    if pd.Timestamp(date).normalize() > expiry:
        next_month = month_start + pd.offsets.MonthBegin(1)
        wednesdays = pd.date_range(next_month, next_month + pd.offsets.MonthEnd(0), freq="W-WED")
        expiry = wednesdays[2]
    return expiry


df["expiry"] = df["date"].apply(nearest_expiry)
df["dte"] = (df["expiry"] - df["date"]).dt.days

df["signal"] = np.where((df["zscore"] >= z_threshold) & (df["dte"] <= max_dte), "研究訊號", "")

latest = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("現貨", f"{latest['spot']:,.2f}")
c2.metric("股期", f"{latest['futures']:,.2f}")
c3.metric("Basis", f"{latest['basis'] * 100:.3f}%")
c4.metric("Z-score", f"{latest['zscore']:.2f}")

st.subheader("基差與 Z-score")
chart_df = df.set_index("date")[["basis", "zscore"]].copy()
chart_df.columns = ["Basis", "Z-score"]
st.line_chart(chart_df)

st.subheader("研究表")
show = df.tail(30)[["date", "spot", "futures", "basis", "zscore", "dte", "signal"]].copy()
show["basis"] = show["basis"] * 100
show = show.rename(columns={"date": "日期", "spot": "現貨", "futures": "股期", "basis": "Basis %", "zscore": "Z-score", "dte": "DTE", "signal": "訊號"})
st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()
st.subheader("下一階段")
st.markdown(
    """
    1. 接入 TWSE / TAIFEX 歷史與即時資料。  
    2. 用理論公平價格扣除利率、股利與成本，建立 **Residual Basis**。  
    3. 加入 Bid/Ask、成交量、未平倉量與滑價。  
    4. 建立到期收斂回測：比較 DTE=1/2/3/5/10 天的收斂機率。  
    5. 加入完整績效指標：勝率、平均報酬、Sharpe、最大回撤與 Profit Factor。  
    """
)

st.caption("⚠️ 本頁為量化研究工具原型，不構成投資建議。")
