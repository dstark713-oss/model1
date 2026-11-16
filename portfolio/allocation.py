from __future__ import annotations

import pandas as pd


def equal_weights(tickers_or_n, total_exposure: float) -> pd.Series:
    """
    Принимает либо list[str] тикеров, либо int n.
    Возвращает Series весов, сумма = total_exposure.
    """
    if isinstance(tickers_or_n, int):
        n = tickers_or_n
        tickers = None
    else:
        tickers = list(tickers_or_n)
        n = len(tickers)

    if n <= 0:
        return pd.Series(dtype=float)

    w = total_exposure / n
    if tickers is None:
        return pd.Series([w] * n)
    return pd.Series([w] * n, index=tickers)