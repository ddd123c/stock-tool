# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 05:31:42 2026

@author: DON998
"""

import pandas as pd
import requests
import io
import ssl

# 忽略 SSL 驗證，確保你電腦能抓到
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def generate_stock_csv():
    print("正在連線證交所抓取清單...")
    
    # 上市 + 上櫃網址
    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", # 上市
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"  # 上櫃
    ]
    
    all_data = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            # 抓取網頁
            res = requests.get(url, headers=headers, verify=False)
            res.encoding = 'cp950'
            
            # 解析表格
            dfs = pd.read_html(io.StringIO(res.text))
            df = dfs[0]
            
            # 整理欄位
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            
            # 找出代號欄位
            col_name = df.columns[0]
            
            for item in df[col_name]:
                try:
                    item_str = str(item).strip()
                    code_str = item_str.split()[0]
                    name_str = item_str.split()[1] if len(item_str.split()) > 1 else item_str
                    
                    # 只留 4 碼個股 (過濾權證)
                    if code_str.isdigit() and len(code_str) == 4:
                        all_data.append([code_str, f"{code_str} {name_str}"])
                except:
                    continue
        except Exception as e:
            print(f"錯誤: {e}")

    # 存成 CSV
    if all_data:
        df_save = pd.DataFrame(all_data, columns=['code', 'name'])
        # 去除重複
        df_save = df_save.drop_duplicates(subset=['code'])
        df_save.to_csv('tw_stocks.csv', index=False, encoding='utf-8')
        print(f"成功！已產生 tw_stocks.csv，共有 {len(df_save)} 檔股票。")
    else:
        print("抓取失敗，請檢查網路。")

if __name__ == "__main__":
    generate_stock_csv()
