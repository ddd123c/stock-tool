# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
from chip_data import fetch_all_weekly_chip_rankings
from institution_data import get_institution_streaks
from quant_engine import compute_indicators, technical_score, strategy_flags

st.set_page_config(page_title="台股 Quant Screener V2.7", layout="wide")
st.title("📊 台股 Quant Screener V2.7")
st.caption("200MA 即時技術篩選 ＋ 神秘金字塔每週大股東 ＋ 法人連買 ＋ 新聞報告")

with st.sidebar:
    st.header("⚙️ 200MA 掃描設定")
    universe = st.radio("股票池", ["全台股", "手動輸入"])
    min_vol = st.number_input("5日均量下限（張）", min_value=0, value=1000, step=500)
    strategy_filter = st.selectbox("200MA策略", ["全部", "只看今日突破", "只看近5日突破", "只看回踩再上"])
    codes_text = st.text_area("手動代號", "2330,2603,3017,3605,6446")

@st.cache_data(ttl=3600)
def get_stock_list():
    return pd.read_csv("tw_stocks.csv", dtype={"code": str})

@st.cache_data(ttl=300, show_spinner=False)
def get_prices(tickers):
    # 1 年資料已足夠計算 200MA、20D斜率與近5日突破，減少首次掃描下載量。
    return yf.download(list(tickers), period="1y", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_institution_rankings(codes):
    return get_institution_streaks(list(codes), lookback_days=45)

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_chip_rankings():
    return fetch_all_weekly_chip_rankings()

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
            # 200MA 入選：最近 5 個交易日內曾由下往上突破 200MA。
            # 今天、第 1～5 個交易日內突破都保留；第 6 天起才排除。
            if not x.get("recent_200_breakout", False):
                continue
            flags = strategy_flags(x)
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

# 使用水平選單取代 st.tabs：Streamlit 的 tabs 會讓所有分頁一起執行，
# 造成一開網頁就同時抓 Yahoo、法人、週籌碼；水平選單只執行目前功能，速度會快很多。
section = st.radio(
    "功能",
    ["🚀 200MA 即時量化篩選", "📈 神秘金字塔｜每週大股東", "🏦 法人連續買超", "📰 新聞報告"],
    horizontal=True,
    label_visibility="collapsed",
)

if section == "🚀 200MA 即時量化篩選":
    def _technical_live_panel():
        st.info("這一頁專注技術面：只保留最近 5 個交易日內突破 200MA 的標的。請手動按「🔄 立即掃描」更新，避免自動頻繁抓取造成資料漏抓。")
        c1, c2 = st.columns([1, 4])
        with c1:
            manual_scan = st.button("🔄 立即掃描", type="primary", key="technical_manual_scan")
        with c2:
            st.caption("🖐️ 手動掃描：需要時按「立即掃描」重新抓取價格")
        if manual_scan:
            get_prices.clear()

        try:
            with st.spinner("正在掃描台股技術面資料..."):
                result = scan_technical(stocks, min_vol, strategy_filter)

            if result.empty:
                st.warning("目前沒有符合條件的標的。可把「200MA策略」設為「全部」確認資料正常。")
            else:
                codes = result["代號"].astype(str).tolist()
                with st.spinner("正在補上法人連買與本週大戶變化..."):
                    try:
                        # 使用 30 分鐘快取，不再每 5 分鐘重新抓法人。
                        inst = _cached_institution_rankings(tuple(codes))
                        result = result.merge(inst, on="代號", how="left")
                    except Exception:
                        for col in ["法人連買天數","外資連買天數","投信連買天數","自營商連買天數"]:
                            result[col] = 0
                        for col in ["外資5日累計(張)","投信5日累計(張)","自營商5日累計(張)"]:
                            result[col] = 0.0
                    try:
                        # 週大戶資料 30 分鐘快取，不再每次技術掃描重新抓。
                        chip = _cached_chip_rankings()
                        chip = chip[["代號", "大股東週增減%"]].drop_duplicates("代號")
                        result = result.merge(chip, on="代號", how="left")
                    except Exception:
                        result["大股東週增減%"] = pd.NA

                def _institution_label(r):
                    parts = []
                    for who, col in [("外資","外資連買天數"),("投信","投信連買天數"),("自營商","自營商連買天數")]:
                        v = r.get(col, 0)
                        if pd.notna(v) and v > 0:
                            parts.append(f"{who}{int(v)}日")
                    return "🔥 " + "＋".join(parts) if parts else "—"

                result["法人標記"] = result.apply(_institution_label, axis=1)
                result["大戶週籌碼"] = result["大股東週增減%"].apply(
                    lambda v: "—" if pd.isna(v) else f"{v:+.2f}%"
                )
                result = result.sort_values("技術分", ascending=False)

                st.success(f"找到 {len(result)} 檔符合 200MA 技術條件的標的")
                display_cols = [
                    "代號","股票","技術分","200MA狀態","站上200MA天數",
                    "法人標記","外資連買天數","投信連買天數","自營商連買天數","大戶週籌碼",
                    "收盤","200MA","量比(5日/20日)","200MA斜率20D%","策略"
                ]
                st.dataframe(result[display_cols], use_container_width=True, hide_index=True)
                st.subheader("🏆 Top 20")
                st.dataframe(result[display_cols].head(20), use_container_width=True, hide_index=True)
                st.caption("200MA、法人、週大戶三者獨立；法人與大戶只作旁邊標記，不參與 200MA 篩選或技術分。")
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
            if chip.empty:
                st.warning("沒有抓到資料。")
            else:
                latest = chip["資料週"].iloc[0].strftime("%Y/%m/%d") if hasattr(chip["資料週"].iloc[0], "strftime") else str(chip["資料週"].iloc[0])
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

elif section == "🏦 法人連續買超":
    st.info("外資、投信、自營商分開計算。只要任一法人目前連續買超 ≥ 3 個交易日就上榜；各法人遇到非買超日就重新計算。")
    st.caption("🖐️ 手動更新：只有按下「更新法人排行」才會重新抓資料。")
    update_inst = st.button("🏦 更新法人排行", type="primary", key="update_institution")
    if update_inst:
        _cached_institution_rankings.clear()
    if update_inst:
      try:
        with st.spinner("正在更新全台股法人買賣超資料..."):
            codes = stocks["code"].astype(str).tolist()
            inst = _cached_institution_rankings(tuple(codes))
            show = stocks[["code", "name"]].copy().rename(columns={"code":"代號","name":"股票"})
            show = show.merge(inst, on="代號", how="left")
            for col in ["外資連買天數","投信連買天數","自營商連買天數"]:
                show[col] = show[col].fillna(0).astype(int)

            # 不再用「三大法人合計」判定，三種法人完全分開。
            show = show[
                (show["外資連買天數"] >= 3) |
                (show["投信連買天數"] >= 3) |
                (show["自營商連買天數"] >= 3)
            ].copy()
            show["最強連買"] = show[["外資連買天數","投信連買天數","自營商連買天數"]].max(axis=1)
            show = show.sort_values(
                ["最強連買","外資連買天數","投信連買天數","自營商連買天數"],
                ascending=False
            )

            st.success(f"目前有 {len(show)} 檔股票至少一種法人連續買超 ≥ 3 日")
            display_cols = [
                "代號","股票","外資連買天數","投信連買天數","自營商連買天數","最強連買",
                "外資5日累計(張)","投信5日累計(張)","自營商5日累計(張)"
            ]
            st.dataframe(show[display_cols], use_container_width=True, hide_index=True)
            st.caption("例如：外資 5 日、投信 0 日，仍會上榜；外資與投信絕不合併計算。")
      except Exception as e:
        st.error(f"法人資料更新失敗：{e}")

else:
    st.info("📰 新聞報告獨立於量化篩選。輸入代號後抓 Yahoo Finance 最新新聞，整理成「事件 → 可能影響 → 觀察重點」的小作文。")
    news_code = st.text_input("股票代號", value="2330", key="news_code")
    news_count = st.slider("新聞數量", 3, 10, 5, key="news_count")
    if st.button("📰 產生新聞報告", type="primary", key="news_report"):
        code = news_code.strip().replace(".TW","").replace(".TWO","")
        if not code.isdigit():
            st.error("請輸入純數字台股代號，例如 2330。")
        else:
            try:
                with st.spinner("正在抓取最新新聞..."):
                    items = yf.Ticker(f"{code}.TW").news[:news_count]
                if not items:
                    st.warning("目前沒有抓到新聞。")
                else:
                    headlines = []
                    for item in items:
                        content = item.get("content", item)
                        title = content.get("title") or item.get("title") or "未命名新聞"
                        publisher = content.get("provider", {}).get("displayName") or item.get("publisher") or ""
                        pub = content.get("pubDate") or item.get("providerPublishTime")
                        headlines.append((title, publisher, pub))
                    st.subheader(f"📝 {code} 新聞小作文")
                    st.write(
                        f"近期新聞焦點主要集中在「{headlines[0][0]}」等事件。"
                        "從目前標題可先觀察公司基本面、產業需求、訂單/產品進度與市場預期是否出現變化。"
                        "新聞本身不代表股價必然上漲或下跌，建議搭配 200MA 趨勢、成交量與法人籌碼交叉確認。"
                    )
                    st.subheader("🗞️ 原始新聞")
                    for title, publisher, pub in headlines:
                        st.markdown(f"- **{title}**  {publisher}")
            except Exception as e:
                st.error(f"新聞抓取失敗：{e}")

st.caption("台股代號來源：tw_stocks.csv｜技術資料：Yahoo Finance｜法人資料：TWSE / TPEx｜週籌碼：神秘金字塔")
