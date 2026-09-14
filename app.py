from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="台股股期套利研究平台", page_icon="📈", layout="wide")

st.title("📈 台股個股期貨套利研究平台")
st.caption("研究用途：現貨／股期價差、Cost of Carry、Residual Basis、Z-score、流動性與收斂回測")

# -----------------------------
# Data layer
# -----------------------------
def demo_data(n: int = 260, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    spot = 1000 + np.cumsum(rng.normal(0, 8, n))
    raw_basis = 0.004 + rng.normal(0, 0.0025, n)
    futures = spot * (1 + raw_basis)
    volume = rng.lognormal(mean=10.5, sigma=0.35, size=n)
    return pd.DataFrame({
        "date": dates,
        "spot": spot,
        "futures": futures,
        "volume": volume,
    })


def normalize_upload(uploaded: object) -> pd.DataFrame:
    if uploaded is None:
        return demo_data()
    x = pd.read_csv(uploaded)
    x.columns = [str(c).strip().lower().replace(" ", "_") for c in x.columns]
    aliases = {
        "datetime": "date", "timestamp": "date", "time": "date",
        "close_price": "spot", "spot_close": "spot",
        "futures_close": "futures", "future": "futures",
        "vol": "volume", "qty": "volume",
    }
    x = x.rename(columns={c: aliases.get(c, c) for c in x.columns})
    required = {"date", "spot", "futures"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"CSV 缺少欄位：{', '.join(sorted(missing))}")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for c in ["spot", "futures", "volume"]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["date", "spot", "futures"]).sort_values("date")
    if "volume" not in x:
        x["volume"] = np.nan
    return x.reset_index(drop=True)


# -----------------------------
# Research calculations
# -----------------------------
def fair_value(spot: pd.Series, annual_rate: float, dividend_yield: float, dte: pd.Series) -> pd.Series:
    t = pd.Series(dte, index=spot.index, dtype=float).clip(lower=0) / 365.0
    return pd.Series(spot, dtype=float) * np.exp((annual_rate - dividend_yield) * t)


def nearest_expiry(date: pd.Timestamp) -> pd.Timestamp:
    month_start = pd.Timestamp(date.year, date.month, 1)
    wednesdays = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0), freq="W-WED")
    expiry = wednesdays[2]
    if date.normalize() > expiry:
        next_month = month_start + pd.offsets.MonthBegin(1)
        wednesdays = pd.date_range(next_month, next_month + pd.offsets.MonthEnd(0), freq="W-WED")
        expiry = wednesdays[2]
    return expiry


def add_features(x: pd.DataFrame, window: int, annual_rate: float, dividend_yield: float, fee_bps: float, slippage_bps: float) -> pd.DataFrame:
    x = x.copy()
    x["expiry"] = x["date"].apply(nearest_expiry)
    x["dte"] = (x["expiry"] - x["date"]).dt.days.clip(lower=0)
    x["basis"] = x["futures"] / x["spot"] - 1.0
    x["fair_value"] = fair_value(x["spot"], annual_rate, dividend_yield, x["dte"])
    x["residual_basis"] = x["futures"] / x["fair_value"] - 1.0
    x["rolling_mean"] = x["residual_basis"].rolling(window, min_periods=max(10, window // 3)).mean()
    x["rolling_std"] = x["residual_basis"].rolling(window, min_periods=max(10, window // 3)).std(ddof=1)
    x["zscore"] = (x["residual_basis"] - x["rolling_mean"]) / x["rolling_std"].replace(0, np.nan)
    if x["volume"].notna().any():
        x["volume_baseline"] = x["volume"].rolling(window, min_periods=max(10, window // 3)).median()
        x["liquidity_ratio"] = x["volume"] / x["volume_baseline"].replace(0, np.nan)
    else:
        x["liquidity_ratio"] = np.nan
    x["estimated_roundtrip_cost"] = (fee_bps + slippage_bps) * 2 / 10000
    x["net_edge_proxy"] = x["residual_basis"].abs() - x["estimated_roundtrip_cost"]
    return x


def run_backtest(x: pd.DataFrame, entry_z: float, exit_z: float, max_holding: int, min_edge: float) -> pd.DataFrame:
    trades = []
    open_i = None
    for i, row in x.reset_index(drop=True).iterrows():
        z = row["zscore"]
        if open_i is None and pd.notna(z) and z >= entry_z and row["net_edge_proxy"] >= min_edge:
            open_i = i
            continue
        if open_i is not None:
            held = i - open_i
            should_exit = (pd.notna(z) and z <= exit_z) or held >= max_holding
            if should_exit:
                entry = x.iloc[open_i]
                rb_entry = float(entry["residual_basis"])
                rb_exit = float(row["residual_basis"])
                pnl = rb_entry - rb_exit - float(entry["estimated_roundtrip_cost"])
                trades.append({
                    "entry_date": entry["date"], "exit_date": row["date"],
                    "holding_days": held, "entry_z": entry["zscore"],
                    "entry_residual_basis": rb_entry, "exit_residual_basis": rb_exit,
                    "net_pnl_proxy": pnl, "win": pnl > 0,
                })
                open_i = None
    return pd.DataFrame(trades)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("研究設定")
    uploaded = st.file_uploader("上傳 CSV 歷史資料", type=["csv"])
    use_demo = st.checkbox("使用展示資料", value=uploaded is None)
    st.divider()
    window = st.slider("Z-score 回看視窗", 20, 180, 60, 5)
    annual_rate = st.number_input("年化資金成本 (%)", 0.0, 20.0, 2.0, 0.1) / 100
    dividend_yield = st.number_input("年化股利殖利率 (%)", 0.0, 20.0, 1.0, 0.1) / 100
    fee_bps = st.number_input("單邊費用估計 (bps)", 0.0, 100.0, 5.0, 0.5)
    slippage_bps = st.number_input("單邊滑價估計 (bps)", 0.0, 100.0, 3.0, 0.5)
    st.divider()
    st.subheader("回測參數")
    entry_z = st.number_input("進場 Z-score", 0.5, 5.0, 2.0, 0.1)
    exit_z = st.number_input("出場 Z-score", -1.0, 2.0, 0.5, 0.1)
    max_holding = st.slider("最大持有天數", 1, 60, 10)
    min_edge = st.number_input("最低淨邊際 (%)", 0.0, 10.0, 0.05, 0.01) / 100

try:
    if uploaded is not None and not use_demo:
        raw = normalize_upload(uploaded)
        data_source = "使用者上傳 CSV"
    else:
        raw = demo_data()
        data_source = "展示資料（非真實行情）"
    df = add_features(raw, window, annual_rate, dividend_yield, fee_bps, slippage_bps)
except Exception as exc:
    st.error(f"資料處理失敗：{exc}")
    st.stop()

st.info(f"資料來源：{data_source}。若要做投資決策，請先接入可驗證的 TWSE / TAIFEX 資料並確認契約與成本。")

latest = df.iloc[-1]
positive_signal = bool(pd.notna(latest["zscore"]) and latest["zscore"] >= entry_z and latest["net_edge_proxy"] >= min_edge)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("現貨", f"{latest['spot']:,.2f}")
c2.metric("股期", f"{latest['futures']:,.2f}")
c3.metric("Fair Value", f"{latest['fair_value']:,.2f}")
c4.metric("Residual Basis", f"{latest['residual_basis'] * 100:.3f}%")
c5.metric("Z-score", "N/A" if pd.isna(latest["zscore"]) else f"{latest['zscore']:.2f}")

if positive_signal:
    st.warning("研究條件觸發：正向偏離。這不是下單建議，請檢查 Bid/Ask、契約乘數、股利、保證金與可借券。")
else:
    st.success("目前沒有符合設定的正向研究訊號。")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 市場 Scanner", "🔍 個股分析", "⏳ 到期收斂", "🧪 回測"])

with tab1:
    st.subheader("市場 Scanner")
    show = df.tail(50)[["date", "spot", "futures", "fair_value", "basis", "residual_basis", "zscore", "dte", "liquidity_ratio", "net_edge_proxy"]].copy()
    show["research_signal"] = np.where(
        (show["zscore"] >= entry_z) & (show["net_edge_proxy"] >= min_edge), "WATCH", "NEUTRAL"
    )
    show = show.rename(columns={
        "date": "日期", "spot": "現貨", "futures": "股期", "fair_value": "Fair Value",
        "basis": "Basis", "residual_basis": "Residual Basis", "zscore": "Z-score",
        "dte": "DTE", "liquidity_ratio": "流動性倍數", "net_edge_proxy": "淨邊際估計",
        "research_signal": "訊號",
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("現貨、股期與公平價值")
    chart = df.set_index("date")[["spot", "futures", "fair_value"]].rename(columns={"spot": "現貨", "futures": "股期", "fair_value": "Fair Value"})
    st.line_chart(chart)
    st.subheader("Residual Basis 與 Z-score")
    chart2 = df.set_index("date")[["residual_basis", "zscore"]].rename(columns={"residual_basis": "Residual Basis", "zscore": "Z-score"})
    st.line_chart(chart2)
    st.write("最新資料列")
    st.dataframe(df.tail(1), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("到期收斂研究")
    st.caption("此頁以簡化的基差收斂代理值做事件研究；正式版本應使用逐筆或至少日內 Bid/Ask、正確契約到期日與實際結算規則。")
    events = []
    for dte in [1, 2, 3, 5, 10, 20]:
        subset = df[df["dte"] <= dte].copy()
        if len(subset) < 2:
            continue
        events.append({
            "DTE 範圍": f"<= {dte}",
            "樣本數": len(subset),
            "平均絕對 Residual Basis (%)": subset["residual_basis"].abs().mean() * 100,
            "正向極端比例 (%)": (subset["zscore"] >= entry_z).mean() * 100,
        })
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    else:
        st.warning("資料不足，無法做 DTE 分組統計。")

with tab4:
    st.subheader("成本調整後回測")
    trades = run_backtest(df, entry_z, exit_z, max_holding, min_edge)
    if trades.empty:
        st.warning("目前沒有完成的交易事件。請增加資料長度或放寬研究條件。")
    else:
        win_rate = trades["win"].mean()
        avg_pnl = trades["net_pnl_proxy"].mean()
        total_pnl = trades["net_pnl_proxy"].sum()
        wins = trades.loc[trades["net_pnl_proxy"] > 0, "net_pnl_proxy"].sum()
        losses = -trades.loc[trades["net_pnl_proxy"] < 0, "net_pnl_proxy"].sum()
        profit_factor = wins / losses if losses > 0 else np.nan
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("交易次數", len(trades))
        m2.metric("勝率", f"{win_rate * 100:.1f}%")
        m3.metric("平均淨邊際代理", f"{avg_pnl * 100:.3f}%")
        m4.metric("Profit Factor", "N/A" if pd.isna(profit_factor) else f"{profit_factor:.2f}")
        equity = trades["net_pnl_proxy"].cumsum()
        st.line_chart(equity.rename("累積淨邊際代理"))
        st.dataframe(trades, use_container_width=True, hide_index=True)

st.divider()
st.caption("⚠️ 本工具為研究原型，不構成投資建議。展示資料、簡化成本與代理收益不可用於實盤績效推論。")
