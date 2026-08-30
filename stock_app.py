import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import urllib3
from io import StringIO
from quant_engine import calculate_200ma_signal

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="台股 200MA 即時量化篩選", layout="wide")

st.title("🚀 200MA 即時量化篩選")
st.caption("歷史日線計算 200MA + 台灣交易所即時價格；手動掃描，不自動重抓。")

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
    "4. 不自動每 5 分鐘掃描，避免平台資源被吃光。"
)


@st.cache_data(ttl=86400, show_spinner=False)
def get_all_tickers():
    """取得上市/上櫃股票代號與交易所別。"""
    stock_dict = {}

    urls = [
        ("TW", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "tse"),
        ("TWO", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", "otc"),
    ]

    for _, url, exchange in urls:
        try:
            res = requests.get(url, timeout=15, verify=False)
            res.raise_for_status()
            df = pd.read_html(StringIO(res.text))[0]
            df.columns = df.iloc[0]
            df = df.iloc[1:]

            for item in df[df.columns[0]].dropna():
                item = str(item).strip()
                parts = item.split()
                if not parts:
                    continue
                code = parts[0]
                if code.isdigit() and len(code) == 4:
                    name = " ".join(parts[1:]) if len(parts) > 1 else code
                    stock_dict[code] = {
                        "name": name,
                        "exchange": exchange,
                        "ticker": f"{code}.{ 'TW' if exchange == 'tse' else 'TWO'}",
                    }
        except Exception:
            continue

    return stock_dict


@st.cache_data(ttl=21600, show_spinner=False)
def download_history_chunk(tickers):
    """Yahoo 歷史日線：每個 chunk 快取 6 小時，避免每次按掃描都重抓。"""
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()

    try:
        return yf.download(
            tickers,
            period="1y",
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
            timeout=20,
        )
    except Exception:
        return pd.DataFrame()


def get_close_frame(data, ticker):
    """兼容單檔與多檔 yfinance 回傳格式。"""
    if data is None or data.empty:
        return None

    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(0):
                return None
            df = data[ticker].copy()
        else:
            df = data.copy()

        if "Close" not in df.columns:
            return None

        return df[["Close", "Volume"]].dropna(how="all")
    except Exception:
        return None


def get_realtime_prices(stock_items):
    """
    使用 TWSE MIS 即時行情。
    上市/上櫃可以批次查詢，不再為每檔股票呼叫 Yahoo 即時報價。
    """
    prices = {}
    items = list(stock_items)
    chunk_size = 80

    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        ex_ch = "|".join(
            f"{item['exchange']}_{code}.tw" for code, item in chunk
        )

        url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"

        try:
            res = requests.get(
                url,
                params={"ex_ch": ex_ch, "json": "1", "delay": "0"},
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            res.raise_for_status()
            payload = res.json()

            for row in payload.get("msgArray", []):
                code = str(row.get("c", "")).strip()
                z = str(row.get("z", "")).strip()
                y = str(row.get("y", "")).strip()

                # 盤中 z 是最新成交價；無成交時退回昨收。
                price_text = z if z not in ("", "-", "0") else y

                try:
                    prices[code] = float(price_text)
                except (TypeError, ValueError):
                    continue

        except Exception:
            continue

    return prices


def scan_200ma(stock_dict, target_codes, min_vol):
    results = []
    cache_plot = {}

    chunk_size = 80
    chunks = [
        target_codes[i:i + chunk_size]
        for i in range(0, len(target_codes), chunk_size)
    ]

    status = st.empty()
    progress = st.progress(0.0)

    status.markdown("🟢 **正在取得台灣交易所即時價格...**")

    realtime_items = [(code, stock_dict[code]) for code in target_codes]
    realtime_prices = get_realtime_prices(realtime_items)

    if not realtime_prices:
        status.error("即時價格取得失敗，請稍後再按一次「立即掃描」。")
        progress.empty()
        return [], {}

    for chunk_no, codes in enumerate(chunks, start=1):
        status.markdown(
            f"🟢 **正在計算 200MA：第 {chunk_no}/{len(chunks)} 批 "
            f"（{len(codes)} 檔）...**"
        )

        yf_tickers = tuple(stock_dict[c]["ticker"] for c in codes)
        data = download_history_chunk(yf_tickers)

        for code in codes:
            info = stock_dict[code]
            ticker = info["ticker"]
            price = realtime_prices.get(code)

            if price is None:
                continue

            df = get_close_frame(data, ticker)

            if df is None or len(df) < 205:
                continue

            volume = pd.to_numeric(df["Volume"], errors="coerce").dropna()

            if len(volume) < 5:
                continue

            vol5 = float(volume.tail(5).mean())

            if vol5 < min_vol * 1000:
                continue

            signal = calculate_200ma_signal(df, price)

            # signal=None 包含「跌破 200MA 就排除」。
            if signal is None:
                continue

            results.append({
                "代號": code,
                "股票": info["name"],
                "現價": round(signal["現價"], 2),
                "200MA": round(signal["200MA"], 2),
                "距200MA": f'{signal["距200MA"]:.2f}%',
                "狀態": signal["突破狀態"],
                "5日均量(張)": int(vol5 / 1000),
            })

            cache_plot[f"{code} {info['name']}"] = df

        progress.progress(chunk_no / max(len(chunks), 1))

    progress.empty()
    status.markdown("🟢 **200MA 掃描完成！**")

    return results, cache_plot


st.markdown(
    """
    <div style="
        padding:16px 20px;
        border-radius:12px;
        background:#18263a;
        margin-bottom:16px;">
        <b>篩選邏輯</b><br>
        現價 > 200MA 才能上榜；最近 5 個交易日內突破者保留；
        一旦跌破 200MA 就排除。
    </div>
    """,
    unsafe_allow_html=True,
)

if source_option == "手動輸入":
    manual_text = st.sidebar.text_area(
        "股票代號",
        "1609,2330,2603,3017,3605,6446",
    )
    target_codes = [
        x.strip()
        for x in manual_text.replace("，", ",").split(",")
        if x.strip()
    ]
else:
    target_codes = None


if st.button("🔄 立即掃描", type="primary"):
    with st.spinner("準備股票清單..."):
        all_stocks = get_all_tickers()

    if not all_stocks:
        st.error("股票清單取得失敗，請稍後再試。")
        st.stop()

    if target_codes is None:
        target_codes = list(all_stocks.keys())
    else:
        target_codes = [code for code in target_codes if code in all_stocks]

    if not target_codes:
        st.warning("沒有找到有效的台股代號。")
        st.stop()

    st.info(
        f"本次掃描 {len(target_codes)} 檔。"
        "歷史資料會快取，下一次掃描不必重新抓一年資料。"
    )

    results, stock_cache = scan_200ma(
        all_stocks,
        target_codes,
        min_vol_limit,
    )

    if results:
        df_result = pd.DataFrame(results)

        # 突破優先，再依距離 200MA 由高到低。
        df_result["_priority"] = (
            df_result["狀態"].str.contains("突破").astype(int)
        )
        df_result["_distance_num"] = (
            df_result["距200MA"].str.replace("%", "", regex=False).astype(float)
        )
        df_result = (
            df_result
            .sort_values(
                ["_priority", "_distance_num"],
                ascending=[False, False]
            )
            .drop(columns=["_priority", "_distance_num"])
            .reset_index(drop=True)
        )

        st.success(f"找到 {len(df_result)} 檔站上 200MA 的股票")

        st.dataframe(
            df_result,
            use_container_width=True,
            hide_index=True,
        )

        if stock_cache:
            selected = st.selectbox(
                "📈 個股歷史走勢",
                list(stock_cache.keys()),
            )

            if selected:
                plot_df = stock_cache[selected].copy()
                plot_df["MA200"] = plot_df["Close"].rolling(200).mean()
                st.line_chart(plot_df[["Close", "MA200"]].tail(250))
    else:
        st.warning(
            "目前沒有符合條件的股票。"
            "注意：跌破 200MA 的股票不會上榜。"
        )
