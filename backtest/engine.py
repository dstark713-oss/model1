from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.costs import costs_from_trades
from portfolio.allocation import equal_weights
from portfolio.breadth import breadth_ratio
from portfolio.correlation import precompute_returns, window_corr, greedy_filter
from portfolio.ranking import compute_score


def _params_for_date(dt: pd.Timestamp, params_by_period: dict) -> dict:
    for (start, end), params in params_by_period.items():
        if start <= dt <= end:
            return params
    return {"n": 4, "threshold": 0.0}


def _trading_calendar(open_by_ticker: dict[str, pd.Series]) -> pd.DatetimeIndex:
    all_dates = set()
    for s in open_by_ticker.values():
        all_dates |= set(s.index)
    return pd.DatetimeIndex(sorted(all_dates))


def _next_trading_date(dates: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp | None:
    idx = dates.searchsorted(dt, side="right")
    return dates[idx] if idx < len(dates) else None


def _compute_pnl(weights_df: pd.DataFrame,
                 open_by_ticker: dict[str, pd.Series],
                 trading_dates: pd.DatetimeIndex) -> pd.Series:
    out = []
    for dt in weights_df.index:
        t1 = _next_trading_date(trading_dates, dt)
        t2 = _next_trading_date(trading_dates, t1) if t1 is not None else None
        w = pd.Series(weights_df.loc[dt, "target_w"])
        if t1 is None or t2 is None or w.empty:
            out.append(0.0)
            continue
        r = 0.0
        for ticker, wt in w.items():
            s = open_by_ticker.get(ticker)
            if s is None:
                continue
            o1 = s.get(t1)
            o2 = s.get(t2)
            if o1 is None or o2 is None or o1 <= 0:
                continue
            r += wt * (o2 / o1 - 1.0)
        out.append(float(r))
    return pd.Series(out, index=weights_df.index, name="ret_gross")


def run_backtest(preds: pd.DataFrame,
                 open_by_ticker: dict[str, pd.Series],
                 close_by_ticker: dict[str, pd.Series],
                 params_by_period: dict,
                 pcfg: dict,
                 bcfg: dict) -> pd.DataFrame:
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    preds["score"] = compute_score(preds["p_tp"], preds["p_sl"])

    dates = sorted(preds["date"].unique())
    if not dates:
        return pd.DataFrame()

    b = breadth_ratio(close_by_ticker, pcfg["allocation"]["breadth_ema"])
    returns_full = precompute_returns(close_by_ticker)
    trading_dates = _trading_calendar(open_by_ticker)

    entry_bps = bcfg["costs"]["entry_bps"]
    exit_bps = bcfg["costs"]["exit_bps"]

    current_w = pd.Series(dtype=float)
    rows = []

    for dt in dates:
        dt = pd.Timestamp(dt)
        day = preds[preds["date"] == dt]
        if day.empty:
            continue

        params = _params_for_date(dt, params_by_period)
        n_star = params["n"]
        th = params["threshold"]

        cand = day.sort_values("score", ascending=False).set_index("ticker")[["score"]]
        cand = cand[cand["score"] > th]

        if cand.empty:
            target_w = pd.Series(dtype=float)
        else:
            corr = window_corr(returns_full, dt, pcfg["correlation"]["window"])
            sel = greedy_filter(cand, corr,
                                threshold=pcfg["correlation"]["threshold"],
                                n_max=n_star)
            exposure = float(b.get(dt, 0.0)) if not b.empty else 0.0
            target_w = equal_weights(sel, exposure)

        cost, turn = costs_from_trades(current_w, target_w, entry_bps, exit_bps)

        rows.append({
            "date": dt,
            "target_w": target_w.to_dict(),
            "turnover": turn,
            "costs": cost,
            "exposure": float(target_w.sum()) if len(target_w) else 0.0,
            "cash": 1.0 - float(target_w.sum()) if len(target_w) else 1.0,
        })
        current_w = target_w

    weights_df = pd.DataFrame(rows).set_index("date").sort_index()
    if weights_df.empty:
        return weights_df

    ret_gross = _compute_pnl(weights_df, open_by_ticker, trading_dates)
    weights_df["ret_gross"] = ret_gross
    weights_df["ret_net"] = weights_df["ret_gross"] - weights_df["costs"]
    return weights_df