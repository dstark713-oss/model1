from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def add_ema_features(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for p in periods:
        e = ema(close, p)
        out[f"ema_{p}"] = e
        out[f"dist_ema_{p}"] = (close - e) / e
        out[f"close_above_ema_{p}"] = (close > e).astype("Int8")
        out[f"ema_{p}_slope"] = e.diff()
    return out


def add_momentum(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    out = df.copy()
    for p in periods:
        out[f"mom_{p}"] = out["close"].pct_change(p)
    return out


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def _adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx},
                        index=df.index)


def add_adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    out = df.copy()
    adx_df = _adx(out, period)
    for c in adx_df.columns:
        out[c] = adx_df[c]
    return out


def add_trend_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = add_ema_features(df, cfg["ema_periods"])
    out = add_momentum(out, cfg["momentum_periods"])
    out = add_adx(out, cfg["adx_period"])
    return out