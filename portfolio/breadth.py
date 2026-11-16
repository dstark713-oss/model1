from __future__ import annotations

import pandas as pd


def breadth_ratio(close_by_ticker: dict[str, pd.Series], ema_period: int = 20) -> pd.Series:
    """
    BreadthRatio_t = (# тикеров с close > EMA20) / (# тикеров с данными на t).
    Возвращает Series по датам.
    """
    flags = []
    for t, s in close_by_ticker.items():
        if s.empty:
            continue
        e = s.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
        flags.append((s > e).rename(t))
    if not flags:
        return pd.Series(dtype=float)
    df = pd.concat(flags, axis=1)
    n = df.notna().sum(axis=1)
    k = df.sum(axis=1, min_count=1)
    return (k / n.replace(0, pd.NA)).rename("breadth_ratio")