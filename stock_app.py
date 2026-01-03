import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 網頁設定 ---
st.set_page_config(page_title="專業操盤手選股 (旗艦版)", layout="wide")
st.title("🤖 台股全自動掃描：多策略戰情室 (旗艦版)")
st.markdown("""
**策略總覽：**
1. **🛡️ 假跌破翻揚**：5日內站上 200MA (標記新入選)。
2. **📈 強勢回調**：多頭排列 + 回測 **15MA** (顯示 20MA 乖離)。
3. **💥 布林突破**：布林壓縮 + 帶量突破 (顯示 20MA 乖離)。
4. **🚀 糾結突破 (新)**：均線糾結 + 漲幅 > 4% + 爆量 (圖片策略)。
""")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 掃描參數")

# 1. 股票來源
st.sidebar.subheader("1. 股票池")
source_option = st.sidebar.radio(
    "掃描範圍：",
    ("全台股 (上市+上櫃)", "手動輸入代號")
)

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
def get_tw_stocks_from_web():
    """爬取上市櫃代號並過濾"""
    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", # 上市
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"  # 上櫃
    ]
    code_list = []
    try:
        for url in urls:
            dfs = pd.read_html(url, encoding='cp950')
            df = dfs[0]
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            col_name = df.columns[0]
            for item in df[col_name]:
                try:
                    code_str = str(item).split()[0]
                    if code_str.isdigit() and len(code_str) == 4:
                        code_list.append(code_str)
                except:
                    continue
        return list(set(code_list))
    except Exception as e:
        st.error(f"清單抓取失敗: {e}")
        return []

def get_target_tickers(source_type, manual_input):
    if source_type == "手動輸入代號":
        manual_input = manual_input.replace("\n", ",").replace(" ", ",")
        return [t.strip() for t in manual_input.split(',') if t.strip()]
    else:
        return get_tw_stocks_from_web()

def calculate_indicators(df):
    """計算技術指標"""
    # 均線系統
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA15'] = df['Close'].rolling(window=15).mean() # 策略2改用 15MA
    df['MA20'] = df['Close'].rolling(window=20).mean() # 月線 (乖離率基準)
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

def analyze_stock(ticker, days, min_vol_zhang):
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
        
        # 共同數據：20MA 乖離率 (所有策略都用這個)
        bias_20 = (curr['Close'] - curr['MA20']) / curr['MA20'] * 100
        
        results = {}
        
        # --- 策略 1: 200MA 假跌破 (5日窗口標記版) ---
        # 邏輯：
        # 1. 檢查「過去 5 個交易日」(包含今天)
        # 2. 哪一天發生了「黃金交叉」(昨天在下，今天在上)
        # 3. 標記是「今天」還是「3天前」
        
        s1_status = None
        # 取最後 6 天數據 (因為要比較前一天，所以需要 5+1 天)
        last_6_days = df.iloc[-6:]
        
        # 從今天往回推 (Index: -1 是今天, -2 是昨天...)
        found_crossover = False
        days_ago_found = -1
        
        # 檢查順序：今天(-1) -> 昨天(-2) -> ... -> 4天前(-5)
        # i 代表倒數第幾天，0是今天，1是昨天...
        for i in range(5): 
            # 今天的 index 是 -1-i
            # 昨天的 index 是 -2-i
            day_curr = last_6_days.iloc[-1-i]
            day_prev = last_6_days.iloc[-2-i]
            
            # 判斷交叉：當天收盤 > 200MA 且 前一天收盤 < 200MA
            if day_curr['Close'] > day_curr['MA200'] and day_prev['Close'] < day_prev['MA200']:
                found_crossover = True
                days_ago_found = i
                break # 找到最近的一次就停止
        
        # 如果有發生交叉，且目前股價還在 200MA 之上 (確保沒跌回去)
        if found_crossover and curr['Close'] > curr['MA200']:
            if days_ago_found == 0:
                s1_status = "🔥 今天入選"
            else:
                s1_status = f"📅 {days_ago_found} 天前入選"
            results['strat_1'] = s1_status

        # --- 策略 2: 強勢回調 (改 15MA + 20MA乖離) ---
        # 1. 多頭排列：15MA > 60MA > 200MA
        # 2. 回測：股價距離 15MA 很近 (例如 3% 內)
        cond2_trend = (curr['MA15'] > curr['MA60']) and (curr['MA60'] > curr['MA200'])
        dist_15 = abs(curr['Close'] - curr['MA15']) / curr['MA15']
        cond2_pullback = (dist_15 < 0.03) and (curr['Close'] > curr['MA60'])
        
        if cond2_trend and cond2_pullback:
            results['strat_2'] = True
            
        # --- 策略 3: 布林突破 (20MA乖離) ---
        if (df['BB_Width'].iloc[-5:-1].mean() < 0.15) and (curr['Close'] > curr['BB_Upper']) and (curr['Volume'] > curr['Vol_MA5']*1.2):
            results['strat_3'] = True

        # --- 策略 4: 均線糾結突破 (圖片策略) ---
        # 1. 均線糾結：MA5, MA10, MA20 非常接近
        # 2. 漲幅 > 4%
        # 3. 總量 > 門檻 (已在前面過濾)
        # 4. 突破：收盤價 > 所有均線
        
        ma_list = [curr['MA5'], curr['MA10'], curr['MA20']]
        ma_max = max(ma_list)
        ma_min = min(ma_list)
        
        # 糾結定義：最高均線和最低均線差距 < 5%
        is_entangled = (ma_max - ma_min) / ma_min < 0.05
        
        # 漲幅計算 (今日收盤 - 昨日收盤) / 昨日收盤
        prev_close = df.iloc[-2]['Close']
        pct_change = (curr['Close'] - prev_close) / prev_close * 100
        
        is_breakout = (curr['Close'] > ma_max) and (pct_change > 4)
        
        if is_entangled and is_breakout:
            results['strat_4'] = True
            
        if not results: return None
        
        return {
            "代號": ticker,
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
    
    with st.spinner("正在掃描全市場... (請耐心等候)"):
        target_tickers = get_target_tickers(source_option, ticker_input)
    
    if not target_tickers:
        st.error("清單抓取失敗。")
    else:
        st.info(f"目標 {len(target_tickers)} 檔，門檻 {min_vol_limit} 張。")
        
        res_s1, res_s2, res_s3, res_s4 = [], [], [], []
        stock_cache = {}
        
        my_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(target_tickers):
            status_text.text(f"分析中 ({i+1}/{len(target_tickers)}): {ticker}")
            my_bar.progress((i+1)/len(target_tickers))
            
            res = analyze_stock(ticker, lookback_days, min_vol_limit)
            if res:
                stock_cache[ticker] = res['df']
                base_info = {
                    "代號": ticker, "收盤": res['收盤'], "均量": res['均量']
                }
                
                # 處理乖離率燈號 (策略2,3,4 通用)
                bias = res['20MA乖離']
                if 3 <= bias <= 8: bias_str = f"✅ {bias}% (完美)"
                elif bias > 10: bias_str = f"⚠️ {bias}% (過熱)"
                elif bias < 0: bias_str = f"🥶 {bias}% (均線下)" # 策略2可能會出現
                else: bias_str = f"{bias}%"

                # 策略 1: 假跌破
                if 'strat_1' in res['策略']:
                    s1 = base_info.copy()
                    s1["200MA"] = res['200MA']
                    s1["入選狀態"] = res['策略']['strat_1']
                    res_s1.append(s1)

                # 策略 2: 強勢回調
                if 'strat_2' in res['策略']:
                    s2 = base_info.copy()
                    s2["20MA乖離"] = bias_str
                    res_s2.append(s2)
                    
                # 策略 3: 布林突破
                if 'strat_3' in res['策略']:
                    s3 = base_info.copy()
                    s3["20MA乖離"] = bias_str
                    res_s3.append(s3)
                    
                # 策略 4: 糾結突破
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
            st.caption("條件：5日內站上 200MA (入選超過 4 天自動隱藏)")
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
            st.caption("條件 (參考圖片)：均線糾結 + 漲幅 > 4% + 帶量突破")
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
