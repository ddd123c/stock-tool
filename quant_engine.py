import pandas as pd
from typing import Optional


def _completed_close(df: pd.DataFrame) -> pd.Series:
    """Return raw (unadjusted) completed trading-day closes only."""
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype="float64")

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return close

    try:
        idx = pd.DatetimeIndex(close.index)
        taipei_today = pd.Timestamp.now(tz="Asia/Taipei").date()

        if idx.tz is not None:
            last_date = idx[-1].tz_convert("Asia/Taipei").date()
        else:
            last_date = idx[-1].date()

        # Today's Yahoo daily bar is incomplete during the session.
        # It must NOT enter the completed-history MA window.
        if last_date == taipei_today:
            close = close.iloc[:-1]
    except Exception:
        pass

    return close


def calculate_200ma_signal(df: pd.DataFrame, realtime_price: Optional[float]):
    """Calculate live, unadjusted 200-trading-day SMA.

    During today's session:
      199 completed trading-day closes + today's latest price = 200 observations.

    Therefore:
      - today's live price is observation #200;
      - on the next trading day, today's final close becomes history;
      - the oldest observation drops out, keeping the window at 200 trading days.

    A stock is NEVER returned when the live price is <= live 200MA.
    """
    if df is None or df.empty or realtime_price is None:
        return None

    completed = _completed_close(df)
    if len(completed) < 199:
        return None

    current_price = float(realtime_price)
    if current_price <= 0:
        return None

    window = completed.tail(199)
    live_ma200 = float((window.sum() + current_price) / 200.0)

    official_ma200 = (
        float(completed.tail(200).mean())
        if len(completed) >= 200 else float("nan")
    )

    cutoff_close = float(window.iloc[0])

    # Find an upward cross within the last 5 completed trading sessions.
    cross_day = None
    if len(completed) >= 201:
        ma200 = completed.rolling(200, min_periods=200).mean()
        first_i = max(200, len(completed) - 5)

        for i in range(len(completed) - 1, first_i - 1, -1):
            prev_ma = ma200.iloc[i - 1]
            cur_ma = ma200.iloc[i]
            if pd.isna(prev_ma) or pd.isna(cur_ma):
                continue

            if completed.iloc[i] > cur_ma and completed.iloc[i - 1] <= prev_ma:
                cross_day = len(completed) - 1 - i
                break

    live_cross = False
    if not pd.isna(official_ma200):
        yesterday_close = float(completed.iloc[-1])
        live_cross = (
            current_price > live_ma200
            and yesterday_close <= official_ma200
        )

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
