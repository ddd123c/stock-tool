import pandas as pd


def calculate_200ma_signal(df: pd.DataFrame, realtime_price: float | None):
    """計算盤中即時 200MA 與最近 5 個交易日突破狀態。

    盤中計算方式：
    - 200MA = 最近 199 個「已完成交易日」收盤價 + 今日即時價，再除以 200。
    - 如果 Yahoo 已經包含今天的日線，就先排除今天那一筆，避免重複計入。
    - 最近 5 個已完成交易日內向上突破者保留。
    - 今日盤中向上突破也保留。
    - 目前跌破 200MA 直接排除。
    """
    if df is None or df.empty or realtime_price is None:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 205:
        return None

    current_price = float(realtime_price)

    # Yahoo 的 1d 資料有時會包含今天的日線，有時只到上一交易日。
    # 盤中 200MA 必須只使用「已完成交易日」的收盤價。
    try:
        last_date = pd.Timestamp(close.index[-1]).date()
        today = pd.Timestamp.now(tz="Asia/Taipei").date()
        if last_date >= today:
            completed_close = close.iloc[:-1]
        else:
            completed_close = close
    except Exception:
        completed_close = close

    if len(completed_close) < 200:
        return None

    # 盤中即時 200MA：199 個已完成交易日 + 今日即時價。
    live_ma200 = float(
        completed_close.tail(199).sum() + current_price
    ) / 200.0

    # 最近 5 個「已完成交易日」是否曾由下往上突破。
    cross_day = None
    ma200_completed = completed_close.rolling(200).mean()

    max_days = min(4, len(completed_close) - 201)
    for days_ago in range(max_days + 1):
        idx = -1 - days_ago
        prev_idx = idx - 1

        if (
            completed_close.iloc[idx] > ma200_completed.iloc[idx]
            and completed_close.iloc[prev_idx] <= ma200_completed.iloc[prev_idx]
        ):
            cross_day = days_ago + 1
            break

    # 今日盤中由下往上突破。
    prev_close = float(completed_close.iloc[-1])
    prev_ma200 = float(ma200_completed.iloc[-1])
    live_cross = current_price > live_ma200 and prev_close <= prev_ma200

    # 最重要：現在跌破 200MA 就不上榜。
    if current_price <= live_ma200:
        return None

    if live_cross:
        status = "🔥 今日突破"
        days = 0
    elif cross_day is not None:
        status = f"📅 {cross_day} 個交易日前突破"
        days = cross_day
    else:
        status = "🟢 站上 200MA"
        days = None

    return {
        "現價": current_price,
        "200MA": live_ma200,
        "距200MA": (current_price - live_ma200) / live_ma200 * 100,
        "突破狀態": status,
        "突破距今交易日": days,
    }
