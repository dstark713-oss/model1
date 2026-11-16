from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sigma(close: pd.Series, vol_window: int, h: int) -> pd.Series:
    """σ_t = Std(returns, vol_window) × sqrt(H), на Close t."""
    ret = close.pct_change()
    return ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(h)


def label_triple_barrier(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    df: date-indexed (или с колонкой date) DataFrame с open/high/low/close.
    Возвращает DataFrame с индексом t (дата сигнала), колонками:
      entry_date, entry, tp, sl, sigma,
      tp_hit (1/0), sl_hit (1/0), timeout (bool), ambiguous (bool)
    Сигнал на Close t, вход на Open t+1, горизонт H сессий.
    """
    h = int(cfg["h"])
    k_tp = float(cfg["k_tp"])
    k_sl = float(cfg["k_sl"])
    vol_window = int(cfg["vol_window"])

    d = df.copy()
    if "date" in d.columns:
        d = d.set_index("date")
    d = d.sort_index()

    close = d["close"]
    sigma = compute_sigma(close, vol_window, h)

    # вход: Open(t+1)
    entry = d["open"].shift(-1)
    entry_date = pd.Series(d.index, index=d.index).shift(-1)

    tp = entry * (1 + k_tp * sigma)
    sl = entry * (1 - k_sl * sigma)

    idx = d.index
    n = len(d)
    high = d["high"].values
    low = d["low"].values
    open_ = d["open"].values
    entry_v = entry.values
    tp_v = tp.values
    sl_v = sl.values

    tp_hit = np.zeros(n, dtype="float")
    sl_hit = np.zeros(n, dtype="float")
    timeout = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(entry_v[i]) or np.isnan(tp_v[i]) or np.isnan(sl_v[i]):
            tp_hit[i] = np.nan
            sl_hit[i] = np.nan
            continue
        # окно t+1 .. t+H (включительно), не выходим за пределы массива
        start = i + 1
        end = min(i + 1 + h, n)
        if start >= n:
            tp_hit[i] = np.nan
            sl_hit[i] = np.nan
            continue

        # фактическая цена входа: если Open(t+1) уже за барьером — entry = Open
        e = entry_v[i]
        tp_lvl = tp_v[i]
        sl_lvl = sl_v[i]

        tp_first = False
        sl_first = False
        amb = False
        to = False

        for j in range(start, end):
            hj = high[j]
            lj = low[j]
            # проверка «открытие за барьером» — уже учтено тем, что entry = Open(t+1),
            # а барьеры считаются от entry. Если же внутри окна Open(t+1) > TP,
            # это значит TP достигнут на открытии — фиксируем как TP first.
            tp_touch = hj >= tp_lvl or (j == start and open_[j] >= tp_lvl)
            sl_touch = lj <= sl_lvl or (j == start and open_[j] <= sl_lvl)

            if tp_touch and sl_touch:
                amb = True
                break
            if tp_touch:
                tp_first = True
                break
            if sl_touch:
                sl_first = True
                break

        if amb:
            ambiguous[i] = True
            tp_hit[i] = np.nan
            sl_hit[i] = np.nan
        elif tp_first:
            tp_hit[i] = 1.0
            sl_hit[i] = 0.0
        elif sl_first:
            tp_hit[i] = 0.0
            sl_hit[i] = 1.0
        else:
            timeout[i] = True
            tp_hit[i] = 0.0
            sl_hit[i] = 0.0

    out = pd.DataFrame({
        "entry_date": entry_date,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "sigma": sigma,
        "tp_hit": tp_hit,
        "sl_hit": sl_hit,
        "timeout": timeout,
        "ambiguous": ambiguous,
    }, index=idx)
    return out