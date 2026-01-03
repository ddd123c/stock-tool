import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import urllib3
from io import StringIO
from datetime import datetime, timedelta

# --- 忽略 SSL 警告 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 網頁設定 ---
st.set_page_config(page_title="專業操盤手選股 (數值修正版)", layout="wide")
st.title("🤖 台股全自動掃描：數值修正極速版")
st.markdown("""
**修正說明：** 已強制設定 `auto_adjust=False`，確保抓取 **原始股價** (非還原權值)，
讓 200MA 與技術指標數值與您的原始程式碼完全一致。
""")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 掃描參數")

# 1. 股票來源
st.sidebar.subheader("1. 股票池")
source_option = st.sidebar.radio(
    "掃描範圍：",
    ("全台股 (上市+上櫃)", "手動輸入代號")
)

# 2. 條件設定
st.sidebar.subheader("2. 篩選條件")
min_vol_limit = st.sidebar.number_input("最小5日均量 (張)", value=2000, step=500)

# --- 核心函數 ---

@st.cache_data
def get_all_tickers():
    """
    抓取台股代號 (上市+上櫃)
    """
    stock_dict = {} 
    
    # 1. 上市
    try:
        url_tw = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res = requests.get(url_tw, verify=False)
        df = pd.read_html(StringIO(res.text))[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        for item in df[df.columns[0]]:
            item = str(item).strip()
            code = item.split()[0]
            if code.isdigit() and len(code) == 4:
                stock_dict[f"{code}.TW"] = item
    except: pass

    # 2. 上櫃
    try:
        url_two = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
        res = requests.get(url_two, verify=False)
        df = pd.read_html(StringIO(res.text))[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        for item in df[df.columns[0]]:
            item = str(item).strip()
            code = item.split()[0]
            if code.isdigit() and len(code) == 4:
                stock_dict[f"{code}.TWO"] = item
    except: pass
        
    return stock_dict

def calculate_indicators_single(df):
    """計算單一股票的技術指標"""
    # 確保資料長度足夠計算 200MA
    if len(df) < 205: return None
    
    # 填補空值
    df = df.ffill()

    # 這裡抓取 'Close' (因為設定了 auto_adjust=False，這就是原始收盤價)
    close = df['Close']
    volume = df['Volume']
    
    # 均線計算
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma15 = close.rolling(15).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()
    
    # 均量
    vol_ma5 = volume.rolling(5).mean()
    
    # 布林通道
    std20 = close.rolling(20).std()
    bb_upper = ma20 + (2 * std20)
    bb_lower = ma20 - (2 * std20)
    
    # 取得最新一筆數據 (iloc[-1])
    c_close = close.iloc[-1]
    c_ma5 = ma5.iloc[-1]
    c_ma15 = ma15.iloc[-1]
    c_ma20 = ma20.iloc[-1]
    c_ma60 = ma60.iloc[-1]
    c_ma200 = ma200.iloc[-1]
    c_vol = volume.iloc[-1]
    c_vol_ma5 = vol_ma5.iloc[-1]
    
    # 取得布林寬度 (前5天平均)
    c_bb_width = (bb_upper.iloc[-5:-1].mean() - bb_lower.iloc[-5:-1].mean()) / ma20.iloc[-5:-1].mean()
    c_bb_upper = bb_upper.iloc[-1]
    
    # 量能過濾 (單位：股 -> 轉張數判斷)
    if pd.isna(c_vol) or (c_vol < min_vol_limit * 1000): 
        return None

    results = {}
    bias_20 = (c_close - c_ma20) / c_ma20 * 100
    
    # --- 策略 1: 假跌破翻揚 (5日內) ---
    found_s1 = False
    days_tag = ""
    # 只有當目前價格 > 200MA 時才檢查過去是否跌破
    if c_close > c_ma200:
        for i in range(5):
            idx = -1 - i
            prev_idx = -2 - i
            # 檢查：當天收盤 > 200MA 且 前一天收盤 < 200MA
            if close.iloc[idx] > ma200.iloc[idx] and close.iloc[prev_idx] < ma200.iloc[prev_idx]:
                found_s1 = True
                days_tag = "🔥 今天入選" if i == 0 else f"📅 {i} 天前入選"
                break
    if found_s1: results['strat_1'] = days_tag
    
    # --- 策略 2: 強勢回調 ---
    cond2_trend = (c_ma15 > c_ma60) and (c_ma60 > c_ma200)
    dist_15 = abs(c_close - c_ma15) / c_ma15
    cond2_pullback = (dist_15 < 0.03) and (c_close > c_ma60)
    if cond2_trend and cond2_pullback: results['strat_2'] = True
    
    # --- 策略 3: 布林突破 ---
    if (c_bb_width < 0.15) and (c_close > c_bb_upper) and (c_vol > c_vol_ma5 * 1.2):
        results['strat_3'] = True
        
    # --- 策略 4: 糾結突破 ---
    ma_list = [c_ma5, ma10.iloc[-1], c_ma20]
    is_entangled = (max(ma_list) - min(ma_list)) / min(ma_list) < 0.05
    prev_close = close.iloc[-2]
    pct_change = (c_close - prev_close) / prev_close * 100
    is_breakout = (c_close > max(ma_list)) and (pct_change > 4)
    if is_entangled and is_breakout: 
        results['strat_4'] = pct_change

    if not results: return None
    
    return {
        "收盤": c_close,
        "200MA": c_ma200,
        "均量": int(c_vol/1000), 
        "20MA乖離": bias_20,
        "策略": results
    }

# --- 主程式 ---

if st.button("🚀 啟動掃描"):
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1. 獲取清單
    with st.spinner("獲取清單中..."):
        all_stocks = get_all_tickers()
    
    if not all_stocks:
        st.error("清單獲取失敗")
        st.stop()
        
    target_tickers = []
    if source_option == "手動輸入代號":
        input_list = [x.strip() for x in st.sidebar.text_area("輸入代號", "2330,2603").split(',')]
        for code in input_list:
            if f"{code}.TW" in all_stocks: target_tickers.append(f"{code}.TW")
            elif f"{code}.TWO" in all_stocks: target_tickers.append(f"{code}.TWO")
            else: target_tickers.append(f"{code}.TW") 
    else:
        target_tickers = list(all_stocks.keys())
    
    st.info(f"掃描標的: {len(target_tickers)} 檔")
    
    res_s1, res_s2, res_s3, res_s4 = [], [], [], []
    stock_cache = {} 
    
    # 設定 Batch 大小
    CHUNK_SIZE = 50
    chunks = [target_tickers[i:i + CHUNK_SIZE] for i in range(0, len(target_tickers), CHUNK_SIZE)]
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"掃描中... {i+1}/{total_chunks}")
        progress_bar.progress((i+1)/total_chunks)
        
        try:
            # [關鍵修正] 加入 auto_adjust=False 確保使用原始股價，與您的原始代碼一致
            data = yf.download(
                chunk, 
                period="2y",     # 抓2年確保 200MA 足夠
                group_by='ticker', 
                auto_adjust=False, # <--- 這是關鍵，不還原權值
                progress=False, 
                threads=True
            )
            
            for ticker in chunk:
                try:
                    if len(chunk) == 1: df = data
                    else: df = data[ticker]
                    
                    if df.empty or df['Close'].isna().all(): continue
                    
                    analysis = calculate_indicators_single(df)
                    
                    if analysis:
                        name = all_stocks.get(ticker, ticker)
                        stock_cache[f"{ticker} {name}"] = df 
                        
                        base_info = {
                            "股票": f"{name}",
                            "代號": ticker,
                            "收盤": float(f"{analysis['收盤']:.2f}"),
                            "均量": analysis['均量']
                        }
                        
                        bias_val = analysis['20MA乖離']
                        if 3 <= bias_val <= 8: bias_str = f"✅ {bias_val:.1f}%"
                        elif bias_val > 10: bias_str = f"⚠️ {bias_val:.1f}%"
                        elif bias_val < 0: bias_str = f"🥶 {bias_val:.1f}%"
                        else: bias_str = f"{bias_val:.1f}%"

                        strat = analysis['策略']
                        
                        if 'strat_1' in strat:
                            row = base_info.copy()
                            row["200MA"] = float(f"{analysis['200MA']:.2f}")
                            row["入選狀態"] = strat['strat_1']
                            res_s1.append(row)
                            
                        if 'strat_2' in strat:
                            row = base_info.copy()
                            row["20MA乖離"] = bias_str
                            res_s2.append(row)
                            
                        if 'strat_3' in strat:
                            row = base_info.copy()
                            row["20MA乖離"] = bias_str
                            res_s3.append(row)
                            
                        if 'strat_4' in strat:
                            row = base_info.copy()
                            row["漲幅%"] = f"🔥 {strat['strat_4']:.2f}%"
                            row["20MA乖離"] = bias_str
                            res_s4.append(row)

                except: continue
        except: continue

    progress_bar.empty()
    status_text.text("✅ 掃描完成！")
    
    t1, t2, t3, t4 = st.tabs(["🛡️ 假跌破 (200MA)", "📈 強勢回調", "💥 布林突破", "🚀 糾結突破"])
    
    def show_table(data_list):
        if data_list: st.dataframe(pd.DataFrame(data_list))
        else: st.warning("無符合")

    with t1: show_table(res_s1)
    with t2: show_table(res_s2)
    with t3: show_table(res_s3)
    with t4: show_table(res_s4)

    st.markdown("---")
    if stock_cache:
        selected = st.selectbox("個股走勢圖", list(stock_cache.keys()))
        if selected:
            df_plot = stock_cache[selected]
            st.line_chart(df_plot['Close'])
