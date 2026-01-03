import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- 網頁設定 ---
st.set_page_config(page_title="專業操盤手選股 (精度修復版)", layout="wide")
st.title("🤖 台股全自動掃描：多策略戰情室 (精度修復版)")
st.markdown("""
**修復：** 解決批次下載時因「非交易日空值」導致 200MA 計算偏差的問題。
**狀態：** ✅ 200MA 精準度已校正。
""")

# --- 1. 讀取 CSV 清單 ---
@st.cache_data
def load_stock_list():
    file_path = 'tw_stocks.csv'
    stock_map = {}
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, dtype=str)
            for index, row in df.iterrows():
                stock_map[row['code']] = row['name']
            return stock_map
        except:
            return {}
    else:
        st.warning("⚠️ 找不到 tw_stocks.csv，請確認已上傳 GitHub。")
        return {'2330': '2330 台積電', '2317': '2317 鴻海', '2603': '2603 長榮'}

all_stock_map = load_stock_list()

# --- 2. 側邊欄參數 ---
st.sidebar.header("⚙️ 掃描參數")
source_option = st.sidebar.radio("掃描範圍：", ("全台股 (讀取 CSV)", "手動輸入代號"))

if source_option == "手動輸入代號":
    default_tickers = "2330, 2317, 2603, 3033, 6116, 2615"
    ticker_input = st.sidebar.text_area("輸入代號 (逗號分隔)", default_tickers)
else:
    ticker_input = ""
    st.sidebar.info(f"已載入 {len(all_stock_map)} 檔股票 (批次處理中)...")

min_vol_limit = st.sidebar.number_input("最小5日均量 (張)", value=2000, step=500)
lookback_days = st.sidebar.slider("資料回溯天數", 300, 600, 400)

# --- 3. 核心指標計算 (精準版) ---
def calculate_indicators(df):
    # --- 關鍵修復：強力清洗無效資料 ---
    # 批次下載時，Yahoo 會塞入很多 NaN 列來對齊日期
    # 我們必須把 'Close' 是 NaN 的列全部丟掉，只留真的有交易的日子
    df = df.dropna(subset=['Close'])
    
    # 確保資料長度足夠算 200MA
    if len(df) < 205: return None
    
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA15'] = df['Close'].rolling(window=15).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    
    low_9 = df['Low'].rolling(window=9).min()
    high_9 = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (2 * std20)
    df['BB_Lower'] = df['MA20'] - (2 * std20)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']
    
    return df

# --- 4. 單檔分析邏輯 ---
def analyze_single_stock(df, code, stock_name, min_vol_zhang):
    try:
        # 計算指標 (內部已經包含 dropna 清洗)
        df = calculate_indicators(df)
        
        # 如果清洗後資料不足，直接返回
        if df is None: return None
        
        # 量能過濾
        avg_vol_shares = df['Volume'].iloc[-5:].mean()
        avg_vol_zhang = avg_vol_shares / 1000
        if avg_vol_zhang < min_vol_zhang: return None
        
        curr = df.iloc[-1]
        
        # 共同數據
        bias_20 = (curr['Close'] - curr['MA20']) / curr['MA20'] * 100
        results = {}
        
        # 策略 1: 假跌破 (5日)
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
            s1_status = "🔥 今天入選" if days_ago_found == 0 else f"📅 {days_ago_found} 天前入選"
            results['strat_1'] = s1_status

        # 策略 2: 強勢回調
        cond2_trend = (curr['MA15'] > curr['MA60']) and (curr['MA60'] > curr['MA200'])
        dist_15 = abs(curr['Close'] - curr['MA15']) / curr['MA15']
        cond2_pullback = (dist_15 < 0.03) and (curr['Close'] > curr['MA60'])
        if cond2_trend and cond2_pullback: results['strat_2'] = True
            
        # 策略 3: 布林突破
        if (df['BB_Width'].iloc[-5:-1].mean() < 0.15) and (curr['Close'] > curr['BB_Upper']) and (curr['Volume'] > curr['Vol_MA5']*1.2):
            results['strat_3'] = True

        # 策略 4: 糾結突破
        ma_list = [curr['MA5'], curr['MA10'], curr['MA20']]
        ma_max = max(ma_list)
        ma_min = min(ma_list)
        is_entangled = (ma_max - ma_min) / ma_min < 0.05
        prev_close = df.iloc[-2]['Close']
        pct_change = (curr['Close'] - prev_close) / prev_close * 100
        is_breakout = (curr['Close'] > ma_max) and (pct_change > 4)
        if is_entangled and is_breakout: results['strat_4'] = True
            
        if not results: return None
        
        return {
            "代號": stock_name, "收盤": float(f"{curr['Close']:.2f}"),
            "200MA": float(f"{curr['MA200']:.2f}"), "20MA乖離": float(f"{bias_20:.2f}"),
            "均量": int(avg_vol_zhang), "漲幅": float(f"{pct_change:.2f}") if 'strat_4' in results else 0.0,
            "策略": results, "df": df
        }
    except Exception:
        return None

# --- 主程式 ---

if st.button("🚀 啟動多策略掃描"):
    
    # 準備清單
    if source_option == "全台股 (讀取 CSV)":
        target_map = all_stock_map
    else:
        manual_input = ticker_input.replace("\n", ",").replace(" ", ",")
        code_list = [t.strip() for t in manual_input.split(',') if t.strip()]
        target_map = {code: all_stock_map.get(code, code) for code in code_list}
    
    if not target_map:
        st.error("清單為空，請檢查 CSV 或輸入內容。")
    else:
        target_codes = list(target_map.keys())
        st.info(f"目標 {len(target_codes)} 檔，採用批次下載模式 (Batch Mode)...")
        
        res_s1, res_s2, res_s3, res_s4 = [], [], [], []
        stock_cache = {}
        
        # --- 批次下載設定 ---
        BATCH_SIZE = 50 
        
        my_bar = st.progress(0)
        status_text = st.empty()
        
        # 開始批次迴圈
        for i in range(0, len(target_codes), BATCH_SIZE):
            batch_codes = target_codes[i : i + BATCH_SIZE]
            batch_symbols = [f"{code}.TW" for code in batch_codes]
            symbols_str = " ".join(batch_symbols)
            
            status_text.text(f"分析進度: {i} / {len(target_codes)} (下載中...)")
            my_bar.progress((i) / len(target_codes))
            
            try:
                # 使用 threads=True 加速下載
                data = yf.download(symbols_str, period="2y", group_by='ticker', threads=True, progress=False)
                
                for code in batch_codes:
                    symbol = f"{code}.TW"
                    stock_name = target_map.get(code, code)
                    
                    try:
                        if len(batch_codes) == 1:
                            df = data
                        else:
                            if symbol not in data.columns.levels[0]:
                                continue 
                            df = data[symbol]
                        
                        if df is None or df.empty: continue

                        # 這裡傳入的是原始 df，會在 analyze_single_stock 內部進行 dropna 清洗
                        res = analyze_single_stock(df.copy(), code, stock_name, min_vol_limit)
                        
                        if res:
                            stock_cache[res['代號']] = res['df']
                            base_info = {"股票": res['代號'], "收盤": res['收盤'], "均量": res['均量']}
                            bias = res['20MA乖離']
                            
                            if 3 <= bias <= 8: bias_str = f"✅ {bias}%"
                            elif bias > 10: bias_str = f"⚠️ {bias}%"
                            elif bias < 0: bias_str = f"🥶 {bias}%"
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
                                
                    except Exception:
                        continue 

            except Exception as e:
                continue 

        my_bar.empty()
        status_text.text("全市場掃描完成！")
        
        t1, t2, t3, t4 = st.tabs(["🛡️ 假跌破 (5日)", "📈 回調 (15MA)", "💥 布林突破", "🚀 糾結突破"])
        
        with t1:
            if res_s1: st.table(pd.DataFrame(res_s1))
            else: st.warning("無符合")
        with t2:
            if res_s2: st.table(pd.DataFrame(res_s2))
            else: st.warning("無符合")
        with t3:
            if res_s3: st.table(pd.DataFrame(res_s3))
            else: st.warning("無符合")
        with t4:
            if res_s4: st.table(pd.DataFrame(res_s4))
            else: st.warning("無符合")
            
        st.markdown("---")
        all_hits = list(stock_cache.keys())
        if all_hits:
            target = st.selectbox("選擇個股查看走勢", all_hits)
            df = stock_cache[target].iloc[-120:]
            st.line_chart(df[['Close', 'MA5', 'MA15', 'MA20', 'MA200']])
