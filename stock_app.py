# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
from chip_data import fetch_all_weekly_chip_rankings
from institution_data import get_institution_streaks
from quant_engine import compute_indicators, technical_score, strategy_flags

st.set_page_config(page_title="台股 Quant Screener V4.6", layout="wide")
st.title("📊 台股 Quant Screener V4.6")
st.caption("200MA 即時技術篩選 ＋ 神秘金字塔每週大股東 ＋ 法人連續買超 ＋ 新聞報告")

with st.sidebar:
    st.header("⚙️ 200MA 掃描設定")
    universe = st.radio("股票池", ["全台股", "手動輸入"])
    min_vol = st.number_input("5日均量下限（張）", min_value=0, value=1000, step=500)
    strategy_filter = st.selectbox("200MA策略", ["全部", "只看今日突破", "只看近5日突破", "只看回踩再上"])
    codes_text = st.text_area("手動代號", "2330,2603,3017,3605,6446")

@st.cache_data(ttl=3600)
def get_stock_list():
    return pd.read_csv("tw_stocks.csv", dtype={"code": str})

@st.cache_data(ttl=21600, show_spinner=False)
def get_history_prices(tickers):
    """Daily trading history for 200SMA; raw Close only, never Adj Close."""
    return yf.download(list(tickers), period="2y", group_by="ticker", auto_adjust=False, actions=True, progress=False, threads=True)

@st.cache_data(ttl=30, show_spinner=False)
def get_live_prices(tickers):
    """Only fetch today's intraday prices; 200MA uses today's latest price."""
    return yf.download(list(tickers), period="1d", interval="5m", group_by="ticker", auto_adjust=False, progress=False, threads=True)

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_chip_rankings():
    return fetch_all_weekly_chip_rankings()

def scan_technical(stocks, min_vol, strategy_filter, progress_slot=None, progress_bar=None):
    tickers = stocks["ticker"].tolist()
    rows = []
    batch_size = 100
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    total = len(batches)

    for batch_no, batch in enumerate(batches, start=1):
        if progress_slot is not None:
            progress_slot.markdown(f":green[🟢 正在掃描第 {batch_no}/{total} 批資料...]")
        if progress_bar is not None:
            progress_bar.progress(batch_no / total, text=f"200MA 掃描進度：{batch_no}/{total} 批")

        try:
            history = get_history_prices(tuple(batch))
            live = get_live_prices(tuple(batch))
        except Exception:
            continue

        for _, stock in stocks[stocks["ticker"].isin(batch)].iterrows():
            code = str(stock["code"])
            ticker = stock["ticker"]
            try:
                if len(batch) == 1:
                    df = history
                    live_df = live
                else:
                    if not hasattr(history, "columns") or ticker not in history.columns.levels[0]:
                        continue
                    df = history[ticker]
                    live_df = live[ticker] if hasattr(live, "columns") and ticker in live.columns.levels[0] else None

                live_price = None
                live_timestamp = None
                if live_df is not None and not live_df.empty and "Close" in live_df.columns:
                    s = pd.to_numeric(live_df["Close"], errors="coerce").dropna()
                    if not s.empty:
                        live_price = float(s.iloc[-1])
                        live_timestamp = s.index[-1]

                x = compute_indicators(df, live_price=live_price, live_timestamp=live_timestamp)
                if not x or x["vol5"] < min_vol * 1000:
                    continue
                if not x.get("above200", False):
                    continue
                if strategy_filter == "只看今日突破" and not x["crossed_up_200"]:
                    continue
                if strategy_filter == "只看近5日突破" and not x["recent_200_breakout"]:
                    continue
                if strategy_filter == "只看回踩再上" and not x["recent_200_retest"]:
                    continue

                flags = strategy_flags(x)
                rows.append({
                    "代號": code, "股票": stock["name"], "技術分": round(technical_score(x), 1),
                    "收盤": round(x["close"], 2), "200MA": round(x["ma200"], 2),
                    "200MA狀態": x["breakout_type"],
                    "突破第幾天": (f"第{x['breakout_day_number']}天" if x.get("breakout_day_number") is not None else "—"),
                    "200MA斜率20D%": round(x["ma200_slope20"] * 100, 2),
                    "量比(5日/20日)": round(x["volume_ratio"], 2),
                    "5日均量(張)": round(x["vol5"] / 1000), "策略": ", ".join(flags.keys())
                })
            except Exception:
                continue

    return pd.DataFrame(rows)

stocks = get_stock_list()
if universe == "手動輸入":
    wanted = {x.strip() for x in codes_text.split(",") if x.strip()}
    stocks = stocks[stocks["code"].isin(wanted)].copy()
stocks["ticker"] = stocks["code"].map(lambda x: f"{x}.TW")

section = st.radio("功能", ["🚀 200MA 即時量化篩選", "📈 神秘金字塔｜每週大股東", "🏛️ 法人連續買超", "📰 新聞報告"], horizontal=True, label_visibility="collapsed")

if section == "🚀 200MA 即時量化篩選":
    def _technical_live_panel():
        st.info("這一頁只抓技術價格資料，不抓法人、不抓大戶、不抓新聞。200MA 按永豐設定：200 個交易日、SMA、原始收盤價；「近5日突破」以突破當天為第1個交易日，第6個交易日開始移除。『全部』顯示目前站上 200MA 的標的；其他策略再依條件篩選。請手動按「🔄 立即掃描」更新。")
        c1, c2 = st.columns([1, 4])
        with c1:
            manual_scan = st.button("🔄 立即掃描", type="primary", key="technical_manual_scan")
        with c2:
            st.caption("🖐️ 手動掃描：需要時按「立即掃描」重新抓取價格")
        if manual_scan:
            get_history_prices.clear()
            get_live_prices.clear()
        try:
            progress_slot = st.empty()
            progress_bar = st.progress(0, text="準備掃描 200MA...")
            result = scan_technical(stocks, min_vol, strategy_filter, progress_slot, progress_bar)
            progress_bar.empty()
            progress_slot.success("🟢 200MA 掃描完成")
            if result.empty:
                st.warning("目前沒有符合條件的標的。可把「200MA策略」設為「全部」確認資料正常。")
            else:
                result = result.sort_values("技術分", ascending=False)
                st.success(f"找到 {len(result)} 檔符合 200MA 技術條件的標的")
                display_cols = ["代號","股票","技術分","200MA狀態","突破第幾天","收盤","200MA","量比(5日/20日)","200MA斜率20D%","策略"]
                st.dataframe(result[display_cols], use_container_width=True, hide_index=True)
                st.subheader("🏆 Top 20")
                st.dataframe(result[display_cols].head(20), use_container_width=True, hide_index=True)
                st.caption("200MA 頁面完全獨立，只使用價格/成交量資料；法人與大戶不會在這裡抓取，也不參與 200MA 篩選或技術分。")
        except Exception as e:
            st.error(f"技術掃描失敗：{e}")
    _technical_live_panel()
elif section == "📈 神秘金字塔｜每週大股東":
    st.info("這一頁專門看神秘金字塔每週籌碼：一次抓全部股票，再分成「增加最多 Top 20」與「減少最多 Top 20」。")
    if st.button("📈 更新本週大股東排行", type="primary"):
        _cached_chip_rankings.clear()
        try:
            with st.spinner("正在抓取神秘金字塔全部股票的最新一週籌碼..."):
                chip = _cached_chip_rankings()
            if chip.empty: st.warning("沒有抓到資料。")
            else:
                latest = chip["資料週"].iloc[0].strftime("%Y/%m/%d") if hasattr(chip["資料週"].iloc[0], "strftime") else str(chip["資料週"].iloc[0])
                st.success(f"共取得 {len(chip)} 檔股票｜最新資料週：{latest}")
                inc = chip.sort_values("大股東週增減%", ascending=False).head(20)
                dec = chip.sort_values("大股東週增減%", ascending=True).head(20)
                left, right = st.columns(2)
                with left:
                    st.subheader("🟢 大股東增加最多 Top 20"); st.dataframe(inc, use_container_width=True, hide_index=True)
                with right:
                    st.subheader("🔴 大股東減少最多 Top 20"); st.dataframe(dec, use_container_width=True, hide_index=True)
                st.caption("排名依神秘金字塔最新一週「>400張大股東持有張數增減率」排序；增加與減少各取 20 檔。")
        except Exception as e: st.error(f"抓取失敗：{e}")
elif section == "🏛️ 法人連續買超":
    st.info("這一頁專門看三大法人連續買超：外資、投信、自營商與合計連買天數。此頁與 200MA 技術掃描完全分開。")
    if st.button("🏛️ 更新法人連續買超", type="primary", key="institution_refresh"):
        try:
            codes = stocks["code"].astype(str).tolist()
            with st.spinner("正在抓取最新法人買賣超資料..."):
                inst = get_institution_streaks(codes)
            if inst.empty:
                st.warning("目前沒有抓到法人資料。")
            else:
                merged = stocks[["code","name"]].copy()
                merged["代號"] = merged["code"].astype(str)
                merged = merged.merge(inst, on="代號", how="left")
                merged = merged.drop(columns=["code"]).rename(columns={"name":"股票"})
                # 只顯示法人買超；法人連買天數 <= 0 的賣超/非買超股票不列入。
                merged = merged[merged["法人連買天數"].fillna(0) > 0].copy()
                merged = merged.sort_values(["法人連買天數","外資5日累計(張)"], ascending=[False,False])
                st.success(f"共取得 {len(merged)} 檔法人連續買超")
                display_cols = ["代號","股票","法人連買天數","外資連買天數","投信連買天數","自營商連買天數","外資5日累計(張)","投信5日累計(張)","自營商5日累計(張)"]
                st.dataframe(merged[display_cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"法人資料抓取失敗：{e}")

else:
    st.info("📰 新聞報告獨立於量化篩選。輸入代號後抓 Yahoo Finance 最新新聞，整理成「事件 → 可能影響 → 觀察重點」的小作文。")
    news_code = st.text_input("股票代號", value="2330", key="news_code")
    news_count = st.slider("新聞數量", 3, 10, 5, key="news_count")
    if st.button("📰 產生新聞報告", type="primary", key="news_report"):
        code = news_code.strip().replace(".TW","").replace(".TWO","")
        if not code.isdigit(): st.error("請輸入純數字台股代號，例如 2330。")
        else:
            try:
                with st.spinner("正在抓取最新新聞..."):
                    items = yf.Ticker(f"{code}.TW").news[:news_count]
                if not items: st.warning("目前沒有抓到新聞。")
                else:
                    headlines = []
                    for item in items:
                        content = item.get("content", item)
                        title = content.get("title") or item.get("title") or "未命名新聞"
                        publisher = content.get("provider", {}).get("displayName") or item.get("publisher") or ""
                        pub = content.get("pubDate") or item.get("providerPublishTime")
                        headlines.append((title,publisher,pub))
                    st.subheader(f"📝 {code} 新聞小作文")
                    st.write(f"近期新聞焦點主要集中在「{headlines[0][0]}」等事件。從目前標題可先觀察公司基本面、產業需求、訂單/產品進度與市場預期是否出現變化。新聞本身不代表股價必然上漲或下跌，建議搭配 200MA 趨勢、成交量與法人籌碼交叉確認。")
                    st.subheader("🗞️ 原始新聞")
                    for title,publisher,pub in headlines: st.markdown(f"- **{title}**  {publisher}")
            except Exception as e: st.error(f"新聞抓取失敗：{e}")
st.caption("台股代號來源：tw_stocks.csv｜技術資料：Yahoo Finance｜週籌碼：神秘金字塔")
