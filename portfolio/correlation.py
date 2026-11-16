from __future__ import annotations

import pandas as pd


def precompute_returns(close_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
    rets = {t: s.pct_change() for t, s in close_by_ticker.items() if not s.empty}
    return pd.DataFrame(rets).sort_index()


def window_corr(returns_full: pd.DataFrame, dt: pd.Timestamp, window: int) -> pd.DataFrame:
    sub = returns_full.loc[:dt].tail(window)
    return sub.corr()


def greedy_filter(candidates: pd.DataFrame, corr: pd.DataFrame,
                  threshold: float, n_max: int) -> list[str]:
    selected: list[str] = []
    if corr.empty:
        return candidates.index.tolist()[:n_max]
    for t in candidates.index:
        if len(selected) >= n_max:
            break
        if t not in corr.columns:
            continue
        ok = True
        for s in selected:
            if s not in corr.columns:
                continue
            c = corr.loc[t, s]
            if pd.notna(c) and abs(c) > threshold:
                ok = False
                break
        if ok:
            selected.append(t)
    return selected