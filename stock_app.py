import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import urllib3 # 新增這個套件用來管理連線警告
from io import StringIO
from datetime import datetime, timedelta

# --- 忽略 SSL 警告 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 網頁設定 ---
st.set_page_config(page_title="專業操盤手選股 (SSL修復版)", layout="wide")
st.title("🤖 台股全自動掃描：多策略戰情室 (SSL修復版)")
st.markdown("""
**策略總覽：**
1. **🛡️ 假跌破翻揚**：5日內站上 200MA (標記新入選)。
2. **📈 強勢回調**：多頭排列 + 回測 **15MA** (顯示 20MA 乖離)。
3. **💥 布林突破**：布林壓縮 + 帶量突破 (顯示 20MA 乖離)。
4. **🚀 糾結突破**：均線糾結 + 漲幅 > 4% + 爆量。
""")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 掃描參數")

# 1. 股票來源
st.sidebar.subheader("1. 股票池")
source_option = st.sidebar.radio(
    "掃描範圍：",
    ("全台股 (上市+上櫃)", "手動輸入代號")
)

# 內建熱門股名稱備份
BACKUP_NAMES = {
    '2330': '2330 台積電', '2317': '2317 鴻海', '2454': '2454 聯發科', '2308': '2308 台達電',
    '2382': '2382 廣達', '2303': '2303 聯電', '2881': '2881 富邦金', '2412': '2412 中華電',
    '2882': '2882 國泰金', '2603': '2603 長榮', '3711': '3711 日月光', '2886': '2886 兆豐金',
    '3231': '3231 緯創', '3008': '3008 大立光', '2609': '2609 陽明', '2615': '2615 萬海',
    '2356': '2356 英業達', '0050': '0050 元大台灣50', '0056': '0056 元大高股息',
    '8069': '8069 元太', '5347': '5347 世界', '6274': '6274 台燿', '3037': '3037 欣興',
    '3034': '3034 聯詠', '2379': '2379 瑞昱', '2345': '2345 智邦', '3035': '3035 智原'
}

if source_option == "手動輸入代號":
    default_tickers = "2330, 2317, 2603, 2356, 3231, 2382, 0050, 8069, 5347, 6274"
    ticker_input = st.sidebar.text_area("輸入代號", default_tickers)
else:
    ticker_input = ""
    st.sidebar.info("系統自動抓取上市櫃清單 (濾除權證)...")

# 2. 條件設定
st.sidebar.subheader("2. 篩選條件")
min_vol_limit = st.sidebar.number_input("最小5日均量 (張)", value=2000, step=500)
lookback_days = st.sidebar.slider("資料回溯天數", 300, 600, 400)

# --- 核心函數 ---

@st.cache_data
def get_tw_stocks_with_names():
    """
    爬取上市櫃代號與名稱 (加入 SSL 忽略與偽裝)
    """
    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", # 上市
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"  # 上櫃
    ]
    stock_map = BACKUP_NAMES.copy()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        for url in urls:
            # 關鍵修改：verify=False (忽略 SSL 憑證檢查)
            response = requests.get(url, headers=headers, verify=False)
            response.encoding = 'cp950'
            
            dfs = pd.read_html(StringIO(response.text))
            df = dfs[0]
            
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            
            col_name = df.columns[0]
            
            for item in df[col_name]:
                try:
                    item_str = str(item).strip()
                    code_str = item_str.split()[0]
                    
                    if code_str.isdigit() and len(code_str) == 4:
                        stock_map[code_str] = item_str
                except:
                    continue
        return stock_map
    except Exception as e:
        st.warning(f"無法連線證交所 (使用內建備份清單): {e}")
        return stock_map

def get_target_tickers(source_type, manual_input):
    all_stock_map = get_tw_stocks_with_names()
    
    if source_type == "手動輸入代號":
        manual_input = manual_input.replace("\n", ",").replace(" ", ",")
        code_list = [t.strip() for t in manual_input.split(',') if t.strip()]
        
        target_map = {}
        for code in code_list:
            target_map[code] = all_stock_map.get(code, code)
        return target_map
        
    else:
        return all_stock_map

def calculate_indicators(df):
    """計算技術指標"""
    # 均線系統
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA15'] = df['Close'].rolling(window=15).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 成交量
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    
    # KD
    low_9 = df['Low'].rolling(window=9).min()
    high_9 = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    
    # 布林通道 (20, 2)
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (2 * std20)
    df['BB_Lower'] = df['MA20'] - (2 * std20)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']
    
    return df

def analyze_stock(ticker, stock_name, days, min_vol_zhang):
    symbol = f"{ticker}.TW"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, multi_level_index=False, auto_adjust=False)
        
        if df.empty or len(df) < 205: return None
        
        # 量能過濾
        avg_vol_shares = df['Volume'].iloc[-5:].mean()
        avg_vol_zhang = avg_vol_shares / 1000
        if avg_vol_zhang < min_vol_zhang: return None
        
        df = calculate_indicators(df)
        
        curr = df.iloc[-1]
        
        # 共同數據：20MA 乖離率
        bias_20 = (curr['Close'] - curr['MA20']) / curr['MA20'] * 100
        
        results = {}
        
        # --- 策略 1: 200MA 假跌破 ---
        s1_status = None
        last_6_days = df.iloc[-6:]
        found_crossover = False
        days_ago_found = -1
        
        for i in range(5): 
            day_curr = last_6_days.iloc[-1-i]
            day_prev = last_6_days.iloc[-2-i]
            if day_curr['Close'] > day_curr['MA200'] and day_prev['Close'] < day_prev['MA200']:
                found_crossover = True
                days_ago_found = i
                break 
        
        if found_crossover and curr['Close'] > curr['MA200']:
            if days_ago_found == 0:
                s1_status = "🔥 今天入選"
            else:
                s1_status = f"📅 {days_ago_found} 天前入選"
            results['strat_1'] = s1_status

        # --- 策略 2: 強勢回調 (15MA) ---
        cond2_trend = (curr['MA15'] > curr['MA60']) and (curr['MA60'] > curr['MA200'])
        dist_15 = abs(curr['Close'] - curr['MA15']) / curr['MA15']
        cond2_pullback = (dist_15 < 0.03) and (curr['Close'] > curr['MA60'])
        
        if cond2_trend and cond2_pullback:
            results['strat_2'] = True
            
        # --- 策略 3: 布林突破 ---
        if (df['BB_Width'].iloc[-5:-1].mean() < 0.15) and (curr['Close'] > curr['BB_Upper']) and (curr['Volume'] > curr['Vol_MA5']*1.2):
            results['strat_3'] = True

        # --- 策略 4: 糾結突破 ---
        ma_list = [curr['MA5'], curr['MA10'], curr['MA20']]
        ma_max = max(ma_list)
        ma_min = min(ma_list)
        is_entangled = (ma_max - ma_min) / ma_min < 0.05
        prev_close = df.iloc[-2]['Close']
        pct_change = (curr['Close'] - prev_close) / prev_close * 100
        is_breakout = (curr['Close'] > ma_max) and (pct_change > 4)
        
        if is_entangled and is_breakout:
            results['strat_4'] = True
            
        if not results: return None
        
        return {
            "代號": stock_name, 
            "Ticker": ticker,   
            "收盤": float(f"{curr['Close']:.2f}"),
            "200MA": float(f"{curr['MA200']:.2f}"),
            "20MA乖離": float(f"{bias_20:.2f}"),
            "均量": int(avg_vol_zhang),
            "漲幅": float(f"{pct_change:.2f}") if 'strat_4' in results else 0.0,
            "策略": results,
            "df": df
        }
    except:
        return None

# --- 主程式 ---

if st.button("🚀 啟動多策略掃描"):
    
    with st.spinner("正在掃描全市場 (已啟用 SSL 修復)..."):
        target_map = get_target_tickers(source_option, ticker_input)
    
    if not target_map:
        st.error("清單抓取失敗。")
    else:
        target_tickers = list(target_map.keys())
        st.info(f"目標 {len(target_tickers)} 檔，門檻 {min_vol_limit} 張。")
        
        res_s1, res_s2, res_s3, res_s4 = [], [], [], []
        stock_cache = {}
        
        my_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(target_tickers):
            stock_name = target_map[ticker]
            
            status_text.text(f"分析中 ({i+1}/{len(target_tickers)}): {stock_name}")
            my_bar.progress((i+1)/len(target_tickers))
            
            res = analyze_stock(ticker, stock_name, lookback_days, min_vol_limit)
            
            if res:
                stock_cache[res['代號']] = res['df']
                
                base_info = {
                    "股票": res['代號'], 
                    "收盤": res['收盤'], 
                    "均量": res['均量']
                }
                
                bias = res['20MA乖離']
                if 3 <= bias <= 8: bias_str = f"✅ {bias}% (完美)"
                elif bias > 10: bias_str = f"⚠️ {bias}% (過熱)"
                elif bias < 0: bias_str = f"🥶 {bias}% (均線下)"
                else: bias_str = f"{bias}%"

                if 'strat_1' in res['策略']:
                    s1 = base_info.copy()
                    s1["200MA"] = res['200MA']
                    s1["入選狀態"] = res['策略']['strat_1']
                    res_s1.append(s1)

                if 'strat_2' in res['策略']:
                    s2 = base_info.copy()
                    s2["20MA乖離"] = bias_str
                    res_s2.append(s2)
                    
                if 'strat_3' in res['策略']:
                    s3 = base_info.copy()
                    s3["20MA乖離"] = bias_str
                    res_s3.append(s3)
                    
                if 'strat_4' in res['策略']:
                    s4 = base_info.copy()
                    s4["漲幅%"] = f"🔥 {res['漲幅']}%"
                    s4["20MA乖離"] = bias_str
                    res_s4.append(s4)
        
        my_bar.empty()
        status_text.text("掃描完成！")
        
        # 顯示結果
        t1, t2, t3, t4 = st.tabs(["🛡️ 假跌破 (5日)", "📈 回調 (15MA)", "💥 布林突破", "🚀 糾結突破"])
        
        with t1:
            st.caption("條件：5日內站上 200MA")
            if res_s1: st.table(pd.DataFrame(res_s1))
            else: st.warning("無符合")
            
        with t2:
            st.caption("條件：回測 15MA + 顯示 20MA 乖離")
            if res_s2: st.table(pd.DataFrame(res_s2))
            else: st.warning("無符合")
            
        with t3:
            st.caption("條件：布林壓縮 + 帶量突破 + 顯示 20MA 乖離")
            if res_s3: st.table(pd.DataFrame(res_s3))
            else: st.warning("無符合")
            
        with t4:
            st.caption("條件：均線糾結 + 漲幅 > 4% + 帶量突破")
            if res_s4: st.table(pd.DataFrame(res_s4))
            else: st.warning("無符合")
            
        # 畫圖區
        st.markdown("---")
        all_hits = list(stock_cache.keys())
        if all_hits:
            target = st.selectbox("選擇個股查看走勢", all_hits)
            df = stock_cache[target].iloc[-120:]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("收盤", f"{df.iloc[-1]['Close']:.2f}")
            c2.metric("15MA", f"{df.iloc[-1]['MA15']:.2f}")
            c3.metric("200MA", f"{df.iloc[-1]['MA200']:.2f}")
            
            st.line_chart(df[['Close', 'MA5', 'MA15', 'MA20', 'MA200']], color=["#FFF", "#0FF", "#FF0", "#F0F", "#F00"])
            st.caption("圖例：白(收盤), 藍(5MA), 黃(15MA), 紫(20MA), 紅(200MA)")
