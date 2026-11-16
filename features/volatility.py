from __future__ import annotations

import numpy as np
import pandas as pd


def add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out[f"atr_{period}"] = tr.ewm(alpha=1 / period, adjust=False,
                                  min_periods=period).mean()
    out[f"atr_pct_{period}"] = out[f"atr_{period}"] / out["close"]
    return out


def add_rv(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    ret = out["close"].pct_change()
    for w in windows:
        out[f"rv_{w}"] = ret.rolling(w, min_periods=w).std() * np.sqrt(252)
    return out


def add_vol_expansion(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    rv_short = out["close"].pct_change().rolling(window, min_periods=window).std()
    rv_long = out["close"].pct_change().rolling(window * 3, min_periods=window * 3).std()
    out["vol_expansion"] = rv_short / rv_long.replace(0, np.nan)
    return out


def add_volatility_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = add_atr(df, cfg["atr_period"])
    out = add_rv(out, cfg["rv_windows"])
    out = add_vol_expansion(out, cfg["vol_exp_window"])
    return out