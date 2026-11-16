from __future__ import annotations

import pandas as pd

from data.storage import SqliteStorage


def load_all_prices(storage: SqliteStorage) -> pd.DataFrame:
    with storage._conn() as c:
        df = pd.read_sql_query(
            "SELECT date, ticker, close, value FROM prices ORDER BY date, ticker", c
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_top100(storage: SqliteStorage,
                   window: int = 60,
                   top_n: int = 100,
                   exclude: set[str] | None = None) -> pd.DataFrame:
    """
    Point-in-time TOP-N по сумме value за последние `window` торговых дней.
    Возвращает long-format DataFrame[date, ticker].
    """
    exclude = exclude or set()
    px = load_all_prices(storage)
    if px.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    if exclude:
        px = px[~px["ticker"].isin(exclude)]

    val = px.pivot_table(index="date", columns="ticker",
                         values="value", aggfunc="sum")
    roll = val.rolling(window=window, min_periods=1).sum()

    out_rows = []
    for dt, row in roll.iterrows():
        s = row.dropna()
        if s.empty:
            continue
        top = s.nlargest(top_n)
        for t in top.index:
            out_rows.append((dt, t))
    return pd.DataFrame(out_rows, columns=["date", "ticker"])


def futures_universe(cfg: dict) -> list[str]:
    """
    Фиксированный список futures-универса:
    Yahoo + MOEX extra (без IMOEX/RGBI — это не торгуемые инструменты).
    """
    tickers = [y["local"] for y in cfg["yahoo"]["tickers"]]
    for e in cfg["moex"]["extra"]:
        if e["local"] in ("IMOEX", "RGBI"):
            continue
        tickers.append(e["local"])
    return tickers