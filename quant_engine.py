import pandas as pd
from typing import Optional


def _completed_close(df: pd.DataFrame) -> pd.Series:
    """只保留已完成的交易日收盤價；不做還原、不使用 Adj Close。"""
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype="float64")

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return close

    # yfinance 盤中可能會先放入今天的未完成日線。
    # 今日價格由交易所即時價提供，因此今天這筆不能再當成歷史收盤。
    try:
        idx = pd.DatetimeIndex(close.index)
        if idx.tz is not None:
            today = pd.Timestamp.now(tz="Asia/Taipei").date()
            last_date = idx[-1].tz_convert("Asia/Taipei").date()
        else:
            today = pd.Timestamp.now(tz="Asia/Taipei").date()
            last_date = idx[-1].date()

        if last_date == today:
            close = close.iloc[:-1]
    except Exception:
        pass

    return close


def calculate_200ma_signal(df: pd.DataFrame, realtime_price: Optional[float]):
    """計算「未還原、200 個交易日」的台股 200MA。

    盤中定義：
      今天 = 第 1 個交易日
      200MA = 前 199 個已完成交易日收盤 + 今天最新成交價，共 200 個值

    因此下個交易日開始後：
      昨天會成為第 1 個完整交易日，
      今天的收盤價也正式進入歷史資料，
      窗口仍維持 200 個交易日。

    扣抵值：
      今天的 200MA 使用「前 199 日 + 今日」；
      到明天時，會被移除的是這 199 日中的最舊那一天。
    """
    if df is None or df.empty or realtime_price is None:
        return None

    completed = _completed_close(df)

    # 至少需要 199 個完整交易日 + 今天即時價。
    if len(completed) < 199:
        return None

    current_price = float(realtime_price)
    if current_price <= 0:
        return None

    # 今天是第 1 天：199 個已完成交易日 + 今日最新價 = 200 個值。
    last_199 = completed.tail(199)
    live_ma200 = float((last_199.sum() + current_price) / 200.0)

    # 最近一個完整交易日的正式 200MA。
    if len(completed) >= 200:
        official_ma200 = float(completed.tail(200).mean())
    else:
        official_ma200 = float("nan")

    # 明天更新時會被移除的是今天窗口中最舊的那一天。
    cutoff_close = float(last_199.iloc[0])

    # 最近 5 個「已完成交易日」內曾由下往上突破。
    cross_day = None
    if len(completed) >= 200:
        ma200_series = completed.rolling(window=200, min_periods=200).mean()
        start = max(200, len(completed) - 5)

        for i in range(len(completed) - 1, start - 1, -1):
            prev_ma = ma200_series.iloc[i - 1]
            cur_ma = ma200_series.iloc[i]

            if pd.isna(prev_ma) or pd.isna(cur_ma):
                continue

            if (
                completed.iloc[i] > cur_ma
                and completed.iloc[i - 1] <= prev_ma
            ):
                cross_day = len(completed) - 1 - i
                break

    # 今日盤中突破：昨日收盤在正式 200MA 下方/等於，今天最新價站上即時 200MA。
    live_cross = False
    if not pd.isna(official_ma200):
        yesterday_close = float(completed.iloc[-1])
        live_cross = (
            current_price > live_ma200
            and yesterday_close <= official_ma200
        )

    # 硬條件：現價跌破或等於盤中 200MA，絕不上榜。
    if current_price <= live_ma200:
        return None

    if live_cross:
        status = "🔥 今日盤中突破"
        days = 0
    elif cross_day is not None:
        status = f"📅 {cross_day} 個交易日前突破"
        days = cross_day
    else:
        status = "🟢 站上 200MA"
        days = None

    # 扣抵方向：用今天會被換掉的最舊值比較。
    if current_price > cutoff_close:
        ma_direction = "⬆️ 向上"
    elif current_price < cutoff_close:
        ma_direction = "⬇️ 向下"
    else:
        ma_direction = "➡️ 持平"

    return {
        "現價": current_price,
        "200MA": live_ma200,
        "正式200MA": official_ma200,
        "扣抵值": cutoff_close,
        "MA方向": ma_direction,
        "距200MA": (current_price - live_ma200) / live_ma200 * 100,
        "突破狀態": status,
        "突破距今交易日": days,
    }
