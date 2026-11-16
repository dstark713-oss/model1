from __future__ import annotations

import pandas as pd


def compute_breadth(prices: dict[str, pd.DataFrame],
                    ema_period: int = 20) -> pd.DataFrame:
    """
    prices: dict[ticker] -> df[date, close]
    Возвращает DataFrame[date, breadth, breadth_ratio] по активному универсу.
    Breadth_i = 1(close_i > EMA20_i). Считается на общем календаре (union дат).
    """
    if not prices:
        return pd.DataFrame(columns=["breadth", "breadth_ratio"])

    frames = []
    for t, df in prices.items():
        if df is None or df.empty:
            continue
        s = df.set_index("date")["close"]
        e = s.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
        above = (s > e).astype("Int8")
        above.name = t
        frames.append(above)
    if not frames:
        return pd.DataFrame(columns=["breadth", "breadth_ratio"])

    all_above = pd.concat(frames, axis=1)
    breadth = all_above.sum(axis=1, min_count=1)
    n = all_above.notna().sum(axis=1)
    ratio = breadth / n.replace(0, pd.NA)
    return pd.DataFrame({"breadth": breadth, "breadth_ratio": ratio})


def compute_position_vs_ema20(prices: dict[str, pd.DataFrame],
                              ema_period: int = 20) -> dict[str, pd.Series]:
    """Для каждого тикера: distance to EMA20 (close - EMA20)/EMA20."""
    out = {}
    for t, df in prices.items():
        if df is None or df.empty:
            continue
        s = df.set_index("date")["close"]
        e = s.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
        out[t] = (s - e) / e
    return out