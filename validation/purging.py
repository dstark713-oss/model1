# validation/purging.py
from __future__ import annotations

import pandas as pd


def purge_split(dates: pd.DatetimeIndex,
                train_end: pd.Timestamp,
                test_start: pd.Timestamp,
                test_end: pd.Timestamp,
                purge_days: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    train: dates <= train_end - purge_days
    test:  test_start <= dates <= test_end
    """
    purge_td = pd.Timedelta(days=purge_days)
    train_mask = dates <= (train_end - purge_td)
    test_mask = (dates >= test_start) & (dates <= test_end)
    return dates[train_mask], dates[test_mask]


def assert_no_overlap(train_dates: pd.DatetimeIndex,
                      test_dates: pd.DatetimeIndex,
                      purge_days: int) -> None:
    if len(train_dates) == 0 or len(test_dates) == 0:
        return
    gap = (test_dates.min() - train_dates.max()).days
    if gap < purge_days:
        raise AssertionError(
            f"purge violated: gap={gap}d < purge_days={purge_days}"
        )