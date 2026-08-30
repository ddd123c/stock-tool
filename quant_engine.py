import pandas as pd


def calculate_200ma_signal(df: pd.DataFrame, realtime_price: float | None):
    """計算 200MA 與最近 5 個交易日突破狀態。

    規則：
    1. 現價必須高於「以即時價格替換今日收盤」的 200MA。
    2. 最近 5 個交易日內突破 200MA 的股票保留。
    3. 如果目前跌破 200MA，直接排除。
    """
    if df is None or df.empty or realtime_price is None:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 205:
        return None

    ma200 = close.rolling(200).mean()
    if pd.isna(ma200.iloc[-1]):
        return None

    current_price = float(realtime_price)

    # 用即時價格取代歷史資料中的最新一筆，得到盤中 200MA。
    hist_before_today = close.iloc[:-1]
    if len(hist_before_today) >= 199:
        live_ma200 = float(
            pd.concat(
                [
                    hist_before_today.tail(199),
                    pd.Series([current_price]),
                ],
                ignore_index=True,
            ).mean()
        )
    else:
        live_ma200 = float(ma200.iloc[-1])

    # 最近 5 個交易日的「由下往上突破」。
    cross_day = None
    max_days = min(4, len(close) - 201)

    for days_ago in range(max_days + 1):
        idx = -1 - days_ago
        prev_idx = idx - 1

        if (
            close.iloc[idx] > ma200.iloc[idx]
            and close.iloc[prev_idx] <= ma200.iloc[prev_idx]
        ):
            cross_day = days_ago
            break

    # 今天盤中由下往上突破。
    prev_close = float(close.iloc[-1])
    prev_ma200 = float(ma200.iloc[-1])
    live_cross = current_price > live_ma200 and prev_close <= prev_ma200

    # 最重要：跌破 200MA 不上榜。
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
