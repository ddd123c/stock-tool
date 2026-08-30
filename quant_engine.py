import pandas as pd


def _completed_close(df: pd.DataFrame) -> pd.Series:
    """只保留已完成的交易日收盤價；rolling 200 即代表 200 個交易日。"""
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return close

    # yfinance 在盤中可能已產生「今天」的日線。
    # 今天尚未收盤，因此不能把它當成正式的第 200 個收盤。
    try:
        idx = pd.DatetimeIndex(close.index)
        if idx.tz is not None:
            today = pd.Timestamp.now(tz="Asia/Taipei").date()
            last_date = idx[-1].tz_convert("Asia/Taipei").date()
        else:
            today = pd.Timestamp.now().date()
            last_date = idx[-1].date()

        if last_date >= today:
            close = close.iloc[:-1]
    except Exception:
        pass

    return close


def calculate_200ma_signal(df: pd.DataFrame, realtime_price: float | None):
    """台股 200MA：200 個交易日窗口 + 扣抵值 + 盤中即時價格。

    正式 200MA：
        最近 200 個「已完成交易日」收盤價平均。

    扣抵概念：
        下一個交易日的 200MA 會移除目前窗口最舊的收盤價，
        再加入新的收盤價。因此：
        新收盤 > 扣抵值 -> 200MA 上升
        新收盤 < 扣抵值 -> 200MA 下降

    盤中：
        用最近 199 個已完成交易日 + 今日即時價，
        計算「今日若以目前價格收盤」的預估 200MA。
    """
    if df is None or df.empty or realtime_price is None:
        return None

    completed = _completed_close(df)
    if len(completed) < 205:
        return None

    current_price = float(realtime_price)
    ma200_series = completed.rolling(window=200, min_periods=200).mean()

    if pd.isna(ma200_series.iloc[-1]):
        return None

    # 正式的最新 200MA（以最後一個已完成交易日為基準）。
    official_ma200 = float(ma200_series.iloc[-1])

    # 扣抵值：下一個交易日更新 200MA 時，會被移除的收盤價。
    cutoff_close = float(completed.iloc[-200])

    # 盤中預估 200MA：
    # 今日尚未收盤，所以用前 199 個已完成交易日 + 今日即時價。
    live_ma200 = float(
        (completed.tail(199).sum() + current_price) / 200.0
    )

    # 最近 5 個「已完成交易日」內是否曾由下往上突破正式 200MA。
    cross_day = None
    start = max(200, len(completed) - 5)

    for i in range(len(completed) - 1, start - 1, -1):
        if i - 1 < 199:
            continue

        prev_ma = ma200_series.iloc[i - 1]
        cur_ma = ma200_series.iloc[i]

        if (
            completed.iloc[i] > cur_ma
            and completed.iloc[i - 1] <= prev_ma
        ):
            cross_day = len(completed) - 1 - i
            break

    # 今日盤中由下往上突破：
    # 昨日收盤 <= 昨日正式 200MA，今天即時價 > 今日預估 200MA。
    yesterday_close = float(completed.iloc[-1])
    yesterday_ma200 = official_ma200
    live_cross = (
        current_price > live_ma200
        and yesterday_close <= yesterday_ma200
    )

    # 硬條件：現在跌破盤中 200MA，絕對不上榜。
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

    # 扣抵方向：用目前即時價格模擬「今天收盤」後，
    # 新價格與明日將扣除的 cutoff 比較。
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
