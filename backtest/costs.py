from __future__ import annotations

import pandas as pd


def turnover(current_w: pd.Series, target_w: pd.Series) -> float:
    """Σ|Δw| по объединению тикеров (вышедшие -> 0)."""
    idx = current_w.index.union(target_w.index)
    c = current_w.reindex(idx, fill_value=0.0)
    t = target_w.reindex(idx, fill_value=0.0)
    return float((t - c).abs().sum())


def costs_from_turnover(turn: float, entry_bps: float, exit_bps: float) -> float:
    """Полные издержки на оборот: покупки + продажи, но не больше оборота."""
    return turn * (entry_bps + exit_bps) / 10000.0

def costs_from_trades(current_w, target_w, entry_bps, exit_bps) -> tuple[float, float]:
    idx = current_w.index.union(target_w.index)
    c = current_w.reindex(idx, fill_value=0.0)
    t = target_w.reindex(idx, fill_value=0.0)
    delta = t - c
    buys = delta.clip(lower=0).sum()
    sells = (-delta).clip(lower=0).sum()
    cost = (buys * entry_bps + sells * exit_bps) / 10000.0
    return float(cost), float(buys + sells)