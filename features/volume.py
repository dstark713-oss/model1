from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _rolling_slope(s: pd.Series, w: int) -> pd.Series:
    arr = s.values.astype(float)
    out = np.full(len(arr), np.nan)
    if len(arr) < w:
        return pd.Series(out, index=s.index)
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    win = sliding_window_view(arr, w)          # (n-w+1, w)
    y_mean = win.mean(axis=1, keepdims=True)
    num = ((x - x_mean) * (win - y_mean)).sum(axis=1)
    out[w-1:] = num / x_var
    return pd.Series(out, index=s.index)


def add_volume_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    vol = out["volume"]

    w = cfg["rel_vol_window"]
    avg = vol.rolling(w, min_periods=w).mean()
    out["rel_volume"] = vol / avg.replace(0, np.nan)
    out["volume_trend"] = _rolling_slope(vol, w)

    w2 = cfg["vol_shock_window"]
    mean = vol.rolling(w2, min_periods=w2).mean()
    std = vol.rolling(w2, min_periods=w2).std()
    out["volume_zscore"] = (vol - mean) / std.replace(0, np.nan)
    out["volume_shock"] = (out["volume_zscore"] >= cfg["vol_shock_z"]).astype("Int8")

    return out