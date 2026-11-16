# signal/updater.py
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import apimoex
import pandas as pd
import requests

from data.storage import SqliteStorage

log = logging.getLogger(__name__)


def _last_date(st: SqliteStorage) -> str | None:
    with st._conn() as c:
        row = c.execute("SELECT MAX(date) FROM prices").fetchone()
    return row[0] if row and row[0] else None


def _tickers(st: SqliteStorage) -> list[str]:
    with st._conn() as c:
        return [r[0] for r in c.execute(
            "SELECT ticker FROM securities ORDER BY ticker"
        ).fetchall()]


def _fetch(session: requests.Session, ticker: str,
           start: str, end: str) -> pd.DataFrame:
    data = apimoex.get_market_candles(
        session, security=ticker, engine="stock", market="shares",
        start=start, end=end, interval=24,
    )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "begin" not in df.columns:
        return pd.DataFrame()
    df["begin"] = pd.to_datetime(df["begin"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"begin": "TRADEDATE"})
    df = df.dropna(subset=["TRADEDATE", "close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="first").sort_values("TRADEDATE")
    out = pd.DataFrame()
    out["TRADEDATE"] = df["TRADEDATE"]
    out["OPEN"] = df.get("open", None)
    out["HIGH"] = df.get("high", None)
    out["LOW"] = df.get("low", None)
    out["CLOSE"] = df["close"]
    out["VOLUME"] = df.get("volume", None)
    return out


def _save(st: SqliteStorage, ticker: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [
        (ticker, r.TRADEDATE,
         float(r.OPEN) if pd.notna(r.OPEN) else None,
         float(r.HIGH) if pd.notna(r.HIGH) else None,
         float(r.LOW) if pd.notna(r.LOW) else None,
         float(r.CLOSE) if pd.notna(r.CLOSE) else None,
         float(r.VOLUME) if pd.notna(r.VOLUME) else None,
         None)
        for r in df.itertuples(index=False)
    ]
    with st._conn() as c:
        c.executemany(
            "INSERT OR IGNORE INTO prices "
            "(ticker,date,open,high,low,close,volume,value) "
            "VALUES (?,?,?,?,?,?,?,?)", rows,
        )
    return len(rows)


def update_to_today(st: SqliteStorage, sleep_between: float = 0.25) -> dict:
    last = _last_date(st)
    today = datetime.now().strftime("%Y-%m-%d")
    log.info("[updater] last=%s today=%s", last, today)
    if last is None:
        return {"status": "empty", "last": None}
    if last >= today:
        log.info("[updater] up to date")
        return {"status": "up_to_date", "last": last}

    start = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = _tickers(st)
    log.info("[updater] range %s..%s, tickers=%d", start, today, len(tickers))

    ok, empty, failed, failed_list = 0, 0, 0, []
    with requests.Session() as s:
        for i, t in enumerate(tickers, 1):
            try:
                df = _fetch(s, t, start, today)
                if df.empty:
                    empty += 1
                else:
                    n = _save(st, t, df)
                    ok += 1
                    log.info("[updater] %d/%d %-10s +%d (%s..%s)",
                             i, len(tickers), t, n,
                             df["TRADEDATE"].iloc[0], df["TRADEDATE"].iloc[-1])
            except Exception as e:
                failed += 1
                failed_list.append(t)
                log.warning("[updater] %d/%d %-10s ERROR %s",
                            i, len(tickers), t, str(e)[:100])
            time.sleep(sleep_between)

    return {
        "status": "done",
        "last_before": last,
        "last_after": _last_date(st),
        "ok": ok, "empty": empty, "failed": failed,
        "failed_list": failed_list,
    }