# -*- coding: utf-8 -*-
"""
台股波段噴發研究室（獨立 Streamlit app）
- 只讀取 tw_stocks.csv；不匯入、不修改既有 app/module。
- 價量資料使用 Yahoo Finance；融資、分點、法人、借券資料可由使用者上傳歷史 CSV。
- 本程式的分數是可解釋的研究排序分數，不是經校準的噴發機率，也不是投資建議。
"""
from __future__ import annotations

import io
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="台股波段噴發研究室", page_icon="📊", layout="wide")

DEFAULT_CODES = "2330, 2449, 3017, 3661, 2308, 2382"


def clean_code(value: object) -> Optional[str]:
    match = re.search(r"(?<!\d)(\d{4,6})(?!\d)", str(value))
    return match.group(1) if match else None


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_list() -> pd.DataFrame:
    try:
        x = pd.read_csv("tw_stocks.csv", dtype=str)
        x.columns = [str(c).strip().lower() for c in x.columns]
        code_col = "code" if "code" in x.columns else x.columns[0]
        name_col = "name" if "name" in x.columns else x.columns[min(1, len(x.columns) - 1)]
        x = x.rename(columns={code_col: "code", name_col: "name"})
        x["code"] = x["code"].map(clean_code)
        return x.dropna(subset=["code"]).drop_duplicates("code").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["code", "name"])


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_history(code: str, period: str = "3y") -> pd.DataFrame:
    """Download one ticker; fall back from TWSE .TW to TPEx .TWO."""
    last_error = None
    for suffix in (".TW", ".TWO"):
        try:
            raw = yf.download(
                f"{code}{suffix}", period=period, interval="1d",
                auto_adjust=False, actions=False, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename_axis("date").reset_index()
            raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
            required = {"date", "close", "high", "low", "volume"}
            if not required.issubset(raw.columns):
                continue
            for col in ("close", "high", "low", "volume"):
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None)
            raw = raw.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
            if len(raw) >= 80:
                raw["code"] = code
                return raw.reset_index(drop=True)
        except Exception as exc:
            last_error = str(exc)
    if last_error:
        raise RuntimeError(f"{code} 下載失敗：{last_error}")
    return pd.DataFrame()


def add_price_features(raw: pd.DataFrame, horizon: int = 20, target_pct: float = 0.15) -> pd.DataFrame:
    """Features at t only use data available at t; label uses t+1 onward."""
    x = raw.copy().sort_values("date").reset_index(drop=True)
    close = x["close"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    volume = x["volume"].fillna(0).astype(float)

    x["ret_5"] = close.pct_change(5)
    x["ret_20"] = close.pct_change(20)
    x["ret_60"] = close.pct_change(60)
    x["ma20"] = close.rolling(20, min_periods=20).mean()
    x["ma60"] = close.rolling(60, min_periods=60).mean()
    x["ma200"] = close.rolling(200, min_periods=150).mean()
    x["ma20_slope_5"] = x["ma20"].pct_change(5)
    x["volume_ma20"] = volume.rolling(20, min_periods=10).mean()
    x["volume_ratio"] = volume / x["volume_ma20"].replace(0, np.nan)
    x["prior_high20"] = high.shift(1).rolling(20, min_periods=15).max()
    x["breakout20"] = close / x["prior_high20"] - 1
    x["range20_pct"] = (
        high.rolling(20, min_periods=15).max() -
        low.rolling(20, min_periods=15).min()
    ) / close.replace(0, np.nan)
    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    x["atr_pct"] = true_range.rolling(14, min_periods=10).mean() / close.replace(0, np.nan)

    # Transparent score, not a probability. Each component is computed using data known at t.
    score = pd.Series(0.0, index=x.index)
    score += (close > x["ma200"]).fillna(False).astype(float) * 15
    score += (close > x["ma20"]).fillna(False).astype(float) * 8
    score += (x["ma20"] > x["ma60"]).fillna(False).astype(float) * 8
    score += (x["ret_20"] > 0).fillna(False).astype(float) * 8
    score += (x["ret_60"] > 0).fillna(False).astype(float) * 8
    score += (x["ma20_slope_5"] > 0).fillna(False).astype(float) * 6
    score += x["breakout20"].between(0, 0.08).fillna(False).astype(float) * 15
    score += x["volume_ratio"].between(1.3, 5.0).fillna(False).astype(float) * 12
    score += (x["ret_5"] > 0).fillna(False).astype(float) * 5
    # Compression is a context feature, not a standalone buy signal.
    range_pct_rank = x["range20_pct"].rolling(120, min_periods=40).rank(pct=True)
    score += (range_pct_rank < 0.35).fillna(False).astype(float) * 8
    # Penalize very extended price; thresholds are configurable research heuristics.
    extension = close / x["ma20"] - 1
    score -= (extension > 0.20).fillna(False).astype(float) * 12
    x["score"] = score.clip(0, 100)

    # Event label: maximum future high over next N sessions; excludes signal day.
    future_high = high.shift(-1)[::-1].rolling(horizon, min_periods=horizon).max()[::-1]
    x["future_max_return"] = future_high / close - 1
    x["event"] = np.where(
        x["future_max_return"].notna(),
        (x["future_max_return"] >= target_pct).astype(float),
        np.nan,
    )
    return x


def normalize_chip_upload(uploaded: object) -> pd.DataFrame:
    raw = pd.read_csv(uploaded)
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    aliases = {
        "日期": "date", "交易日期": "date", "stock_code": "code", "股票代號": "code",
        "股票代號/名稱": "code", "融資餘額": "margin_balance", "融資今日餘額": "margin_balance",
        "分點淨買超": "broker_net_buy", "券商分點淨買超": "broker_net_buy",
        "法人淨買超": "institutional_net_buy", "三大法人淨買超": "institutional_net_buy",
        "借券賣出餘額": "borrow_balance",
    }
    raw = raw.rename(columns={c: aliases.get(c, c) for c in raw.columns})
    required = {"date", "code"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("籌碼 CSV 至少需要 date、code 欄位；缺少：" + ", ".join(sorted(missing)))
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["code"] = raw["code"].map(clean_code)
    raw = raw.dropna(subset=["date", "code"]).sort_values(["code", "date"])
    for col in ("margin_balance", "broker_net_buy", "institutional_net_buy", "borrow_balance"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if "margin_balance" in raw:
        raw["margin_change_5d"] = raw.groupby("code")["margin_balance"].diff(5)
        raw["margin_change_20d"] = raw.groupby("code")["margin_balance"].diff(20)
    if "broker_net_buy" in raw:
        raw["broker_net_5d"] = raw.groupby("code")["broker_net_buy"].transform(lambda s: s.rolling(5, min_periods=1).sum())
        raw["broker_net_20d"] = raw.groupby("code")["broker_net_buy"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    if "institutional_net_buy" in raw:
        raw["institutional_net_5d"] = raw.groupby("code")["institutional_net_buy"].transform(lambda s: s.rolling(5, min_periods=1).sum())
        raw["institutional_net_20d"] = raw.groupby("code")["institutional_net_buy"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    if "borrow_balance" in raw:
        raw["borrow_change_5d"] = raw.groupby("code")["borrow_balance"].diff(5)
    return raw


def latest_chip_snapshot(chips: pd.DataFrame) -> pd.DataFrame:
    if chips.empty:
        return pd.DataFrame()
    derived = [
        "margin_balance", "margin_change_5d", "margin_change_20d",
        "broker_net_buy", "broker_net_5d", "broker_net_20d",
        "institutional_net_buy", "institutional_net_5d", "institutional_net_20d",
        "borrow_balance", "borrow_change_5d",
    ]
    cols = ["code", "date"] + [c for c in derived if c in chips.columns]
    return chips[cols].sort_values("date").groupby("code", as_index=False).tail(1).drop(columns=["date"], errors="ignore")


def make_summary(code: str, name: str, features: pd.DataFrame) -> dict:
    valid = features.dropna(subset=["close"])
    if valid.empty:
        return {}
    row = valid.iloc[-1]
    return {
        "代號": code, "股票": name, "研究分數": round(float(row["score"]), 1),
        "收盤價": round(float(row["close"]), 2),
        "5日報酬%": round(float(row["ret_5"] * 100), 2) if pd.notna(row["ret_5"]) else np.nan,
        "20日報酬%": round(float(row["ret_20"] * 100), 2) if pd.notna(row["ret_20"]) else np.nan,
        "60日報酬%": round(float(row["ret_60"] * 100), 2) if pd.notna(row["ret_60"]) else np.nan,
        "量比(當日/20日均量)": round(float(row["volume_ratio"]), 2) if pd.notna(row["volume_ratio"]) else np.nan,
        "距20日高點%": round(float(row["breakout20"] * 100), 2) if pd.notna(row["breakout20"]) else np.nan,
        "20日區間%": round(float(row["range20_pct"] * 100), 2) if pd.notna(row["range20_pct"]) else np.nan,
        "200MA上方": bool(row["close"] > row["ma200"]) if pd.notna(row["ma200"]) else False,
        "訊號日期": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
    }


def main() -> None:
    st.title("📊 台股波段噴發研究室")
    st.caption("獨立研究 App｜價量候選排序 × 融資／分點／法人資料 × 歷史事件研究")
    st.warning(
        "研究工具而非買賣建議。研究分數不是噴發機率；Yahoo Finance 價量資料可能延遲或有缺漏。"
        "融資、分點、法人及借券欄位只有在上傳相符的歷史 CSV 後才會納入顯示。"
    )

    stock_list = load_stock_list()
    with st.sidebar:
        st.header("① 掃描設定")
        period = st.selectbox("價量歷史期間", ["2y", "3y", "5y"], index=1)
        source_mode = st.radio("股票池", ["手動輸入代號", "從台股清單選股"])
        if source_mode == "手動輸入代號":
            codes_text = st.text_area("股票代號（逗號分隔）", DEFAULT_CODES, height=90)
            codes = list(dict.fromkeys(clean_code(v) for v in codes_text.split(",") if clean_code(v)))
        else:
            options = stock_list["code"].tolist()
            defaults = [c for c in ["2330", "2449", "3017", "3661"] if c in options]
            codes = st.multiselect("選擇股票（可搜尋代號）", options, default=defaults, format_func=lambda c: f"{c}  {stock_list.loc[stock_list['code'].eq(c), 'name'].iloc[0] if stock_list['code'].eq(c).any() else ''}")
        horizon = st.slider("噴發事件觀察期（交易日）", 5, 60, 20, 5)
        target_pct = st.slider("噴發定義：未來期間最高漲幅", 5, 40, 15, 1) / 100
        score_cutoff = st.slider("候選分數門檻", 0, 100, 55, 5)
        st.divider()
        st.header("② 籌碼資料（選填）")
        chip_upload = st.file_uploader("上傳每日籌碼 CSV", type=["csv"], help="需包含 date、code；可選 margin_balance、broker_net_buy、institutional_net_buy、borrow_balance。單位請保持全檔一致。")
        template = pd.DataFrame([{
            "date": "2026-10-01", "code": "2330", "margin_balance": 10000,
            "broker_net_buy": 120, "institutional_net_buy": 500, "borrow_balance": 800,
        }, {
            "date": "2026-10-02", "code": "2330", "margin_balance": 10100,
            "broker_net_buy": -20, "institutional_net_buy": 300, "borrow_balance": 820,
        }])
        st.download_button("下載籌碼 CSV 範本", template.to_csv(index=False).encode("utf-8-sig"), "chip_template.csv", "text/csv")

    if not codes:
        st.info("請先輸入或選擇至少一檔股票。")
        st.stop()

    name_map = {}
    if not stock_list.empty:
        name_map = dict(zip(stock_list["code"], stock_list["name"]))

    chip_df = pd.DataFrame()
    if chip_upload is not None:
        try:
            chip_df = normalize_chip_upload(chip_upload)
            st.sidebar.success(f"已讀取籌碼資料：{len(chip_df):,} 列")
        except Exception as exc:
            st.sidebar.error(f"籌碼 CSV 格式錯誤：{exc}")

    progress = st.progress(0, text="準備下載歷史價量資料…")
    status = st.empty()
    all_features = {}
    summaries = []
    errors = []
    for idx, code in enumerate(codes, start=1):
        status.write(f"正在取得 {code}（{idx}/{len(codes)}）…")
        progress.progress(idx / len(codes), text=f"歷史資料：{idx}/{len(codes)}")
        try:
            raw = fetch_history(code, period)
            if raw.empty:
                errors.append(f"{code}: Yahoo Finance 無可用日線")
                continue
            feat = add_price_features(raw, horizon=horizon, target_pct=target_pct)
            all_features[code] = feat
            summary = make_summary(code, name_map.get(code, code), feat)
            if summary:
                summaries.append(summary)
        except Exception as exc:
            errors.append(f"{code}: {exc}")
        time.sleep(0.05)
    progress.empty()
    status.empty()

    if not summaries:
        st.error("沒有取得可用的歷史資料。請確認代號，或稍後重試。")
        if errors:
            st.code("\n".join(errors))
        st.stop()

    result = pd.DataFrame(summaries).sort_values("研究分數", ascending=False)
    if not chip_df.empty:
        snapshot = latest_chip_snapshot(chip_df)
        if not snapshot.empty:
            result["代號"] = result["代號"].astype(str)
            snapshot["code"] = snapshot["code"].astype(str)
            result = result.merge(snapshot, left_on="代號", right_on="code", how="left").drop(columns=["code"], errors="ignore")
            display_rename = {
                "margin_balance": "最新融資餘額", "margin_change_5d": "融資5日增減",
                "margin_change_20d": "融資20日增減", "broker_net_buy": "分點最新淨買超",
                "broker_net_5d": "分點5日淨買超", "broker_net_20d": "分點20日淨買超",
                "institutional_net_buy": "法人最新淨買超", "institutional_net_5d": "法人5日淨買超",
                "institutional_net_20d": "法人20日淨買超", "borrow_balance": "最新借券餘額",
                "borrow_change_5d": "借券5日增減",
            }
            result = result.rename(columns=display_rename)

    st.subheader("候選股排序")
    st.caption("分數由均線趨勢、動能、20 日突破、量比與區間壓縮等規則加總；不是機器學習機率，也沒有預設分點或融資一定利多。")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button("下載候選股 CSV", result.to_csv(index=False).encode("utf-8-sig"), "breakout_candidates.csv", "text/csv")

    candidates = result[result["研究分數"] >= score_cutoff]
    st.metric("達到分數門檻的候選股", f"{len(candidates)} / {len(result)}")
    if not candidates.empty:
        st.dataframe(candidates, use_container_width=True, hide_index=True)

    st.subheader("歷史事件研究：高分訊號之後，真的比較容易噴發嗎？")
    st.caption(
        f"事件定義：訊號日之後 {horizon} 個交易日內，最高價相對訊號日收盤價上漲至少 {target_pct:.0%}。"
        "僅使用完整觀察期的訊號；這是歷史條件發生率，不代表未來機率。"
    )
    event_frames = []
    for code, feat in all_features.items():
        valid = feat.dropna(subset=["future_max_return", "score"]).copy()
        if valid.empty:
            continue
        valid["code"] = code
        valid["event"] = (valid["future_max_return"] >= target_pct).astype(int)
        event_frames.append(valid[["date", "code", "score", "future_max_return", "event", "ret_20", "volume_ratio", "breakout20"]])
    if event_frames:
        events = pd.concat(event_frames, ignore_index=True)
        if len(events) >= 30:
            base_rate = events["event"].mean()
            high = events[events["score"] >= score_cutoff]
            high_rate = high["event"].mean() if len(high) else np.nan
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("歷史樣本數", f"{len(events):,}")
            m2.metric("全樣本事件率", f"{base_rate:.1%}")
            m3.metric("高分組事件率", f"{high_rate:.1%}" if pd.notna(high_rate) else "樣本不足")
            m4.metric("高分組樣本數", f"{len(high):,}")
            if pd.notna(high_rate) and base_rate > 0:
                st.metric("高分組／全樣本事件率比", f"{high_rate / base_rate:.2f}x")
            bins = pd.cut(events["score"], bins=[-1, 20, 40, 60, 80, 101], labels=["0–20", "21–40", "41–60", "61–80", "81–100"])
            table = events.assign(分數區間=bins).groupby("分數區間", observed=False).agg(
                樣本數=("event", "size"),
                噴發事件數=("event", "sum"),
                事件率=("event", "mean"),
                未來期間最高漲幅平均=("future_max_return", "mean"),
                分數平均=("score", "mean"),
            ).reset_index()
            table["事件率"] = (table["事件率"] * 100).round(2)
            table["未來期間最高漲幅平均"] = (table["未來期間最高漲幅平均"] * 100).round(2)
            table["分數平均"] = table["分數平均"].round(1)
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.bar_chart(table.set_index("分數區間")[["事件率"]])
            st.download_button("下載歷史事件樣本", events.to_csv(index=False).encode("utf-8-sig"), "breakout_event_study.csv", "text/csv")
            st.warning(
                "重要限制：不同日期的觀察窗彼此重疊，同一股票會重複出現；此處屬探索性事件研究。"
                "要聲稱有穩定預測力，還需按時間切分訓練／測試、處理多重檢定與交易成本，並使用完整歷史股票池避免存活者偏誤。"
            )
        else:
            st.info("歷史樣本不足 30 筆，暫不顯示事件率比較；可增加股票數或延長歷史期間。")
    else:
        st.info("目前沒有足夠完整的未來觀察窗可供事件研究。")

    if errors:
        with st.expander(f"資料下載提醒（{len(errors)} 檔）"):
            st.write("\n".join(errors))

    st.subheader("單一股票診斷")
    chosen = st.selectbox("選擇查看走勢", list(all_features.keys()))
    chart = all_features[chosen].tail(260).set_index("date")
    left, right = st.columns(2)
    with left:
        st.markdown("**收盤價與均線**")
        st.line_chart(chart[["close", "ma20", "ma60", "ma200"]])
    with right:
        st.markdown("**研究分數與量比**")
        st.line_chart(chart[["score", "volume_ratio"]])
    details = all_features[chosen].tail(30)[
        ["date", "close", "volume", "ret_5", "ret_20", "ret_60", "volume_ratio", "breakout20", "range20_pct", "score", "future_max_return", "event"]
    ].copy()
    details["date"] = pd.to_datetime(details["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(details, use_container_width=True, hide_index=True)

    st.caption("資料來源：Yahoo Finance 價量資料與使用者上傳的籌碼 CSV。此 App 不會改寫原有 stock_app.py、app.py 或任何既有模組。")


if __name__ == "__main__":
    main()
