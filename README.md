# 台股 Quant Screener V2

以原本的 200MA 篩選器為核心，加入大股東週籌碼與多因子評分。

## V2
- 200MA 站上與 20 日斜率
- 15/60/200MA 多頭排列
- 20MA 乖離、量能、布林壓縮、均線糾結突破
- 神秘金字塔大股東 (>400 張) 每週持有張數成長率
- 1/2/4 週籌碼變化與趨勢
- 技術分 + 籌碼分 = Quant Score
- CSV / 網站資料來源

## 下一階段
1. 上市/上櫃 .TW/.TWO fallback
2. PE/PB/殖利率/EPS/營收/ROE
3. 外資/投信/融資融券
4. 歷史 snapshot database
5. walk-forward backtest、最大回撤、Sharpe、勝率
6. 完整參數化
7. 個股研究頁

回測必須使用當時可取得的資料，避免 look-ahead bias。
