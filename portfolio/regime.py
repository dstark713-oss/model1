from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def compute_regime(imoex_df: pd.DataFrame, ema_period: int = 100) -> pd.Series:
    """
    Close(IMOEX) > EMA(ema_period) -> 'stocks', иначе 'futures'.
    Индекс — date.
    """
    if imoex_df is None or imoex_df.empty:
        return pd.Series(dtype="object", name="regime")
    df = imoex_df.sort_values("date").set_index("date")
    close = pd.to_numeric(df["close"], errors="coerce")
    e = ema(close, ema_period)
    regime = pd.Series("futures", index=close.index, name="regime", dtype="object")
    regime[close > e] = "stocks"
    regime[e.isna()] = None
    return regime