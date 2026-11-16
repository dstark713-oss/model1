from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _stochastic(df: pd.DataFrame, k: int, d: int) -> pd.DataFrame:
    low = df["low"].rolling(k, min_periods=k).min()
    high = df["high"].rolling(k, min_periods=k).max()
    pct_k = 100 * (df["close"] - low) / (high - low).replace(0, np.nan)
    pct_d = pct_k.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"stoch_k": pct_k, "stoch_d": pct_d}, index=df.index)


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": macd, "macd_signal": sig,
                         "macd_hist": macd - sig}, index=close.index)


def add_oscillator_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = _rsi(out["close"], cfg["rsi_period"])
    st = _stochastic(out, cfg["stoch_k"], cfg["stoch_d"])
    for c in st.columns:
        out[c] = st[c]
    m = _macd(out["close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    for c in m.columns:
        out[c] = m[c]
    return out