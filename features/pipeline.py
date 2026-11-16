from __future__ import annotations

import pandas as pd

from features.trend import add_trend_features
from features.volatility import add_volatility_features
from features.volume import add_volume_features
from features.oscillators import add_oscillator_features
from features.price_action import add_price_action_features


def compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out = out.set_index("date").sort_index()
    else:
        out = out.sort_index()
    out = add_trend_features(out, cfg["trend"])
    out = add_volatility_features(out, cfg["volatility"])
    out = add_volume_features(out, cfg["volume"])
    out = add_oscillator_features(out, cfg["oscillators"])
    out = add_price_action_features(out, cfg["price_action"])
    return out