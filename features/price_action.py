from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_action_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    w = cfg["breakout_window"]

    out["rolling_high"] = out["high"].rolling(w, min_periods=w).max()
    out["rolling_low"] = out["low"].rolling(w, min_periods=w).min()
    out["dist_to_high"] = (out["rolling_high"] - out["close"]) / out["close"]
    out["dist_to_low"] = (out["close"] - out["rolling_low"]) / out["close"]
    out["breakout_up"] = (out["close"] >= out["rolling_high"]).astype("Int8")
    out["breakout_down"] = (out["close"] <= out["rolling_low"]).astype("Int8")

    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["body"] = (out["close"] - out["open"]) / rng
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / rng
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / rng

    return out