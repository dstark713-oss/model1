# models/common.py
from __future__ import annotations

import pandas as pd


FEATURE_COLS_TO_DROP = [
    "date", "ticker", "open", "high", "low", "close", "volume", "value",
    "entry_date", "entry", "tp", "sl", "sigma",
    "tp_hit", "sl_hit", "timeout", "ambiguous",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = set(FEATURE_COLS_TO_DROP)
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def time_split(n: int, test_fraction: float, embargo: int) -> tuple[slice, slice]:
    test_start = int(n * (1 - test_fraction))
    train_end = max(0, test_start - embargo)
    return slice(0, train_end), slice(test_start, n)