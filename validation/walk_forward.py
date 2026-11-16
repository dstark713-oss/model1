# validation/walk_forward.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from validation.purging import purge_split


@dataclass
class WFFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=months)


def _add_years(ts: pd.Timestamp, years: int) -> pd.Timestamp:
    return ts + pd.DateOffset(years=years)


def generate_folds(all_dates: pd.DatetimeIndex, cfg: dict) -> list[WFFold]:
    """
    WF-фолды.
    Первый test_start = min(all_dates) + pretrain_years.
    Каждый следующий test_start += retrain_months (== test_months).
    test_end = test_start + test_months (не дальше max(all_dates)).
    train = [train_start, test_start - purge], где train_start:
      - expanding: min(all_dates)
      - rolling:   test_start - rolling_years
    """
    if len(all_dates) == 0:
        return []

    dates = pd.DatetimeIndex(sorted(all_dates))
    min_d = dates.min()
    max_d = dates.max()

    test_months = int(cfg["test_months"])
    retrain_months = int(cfg["retrain_months"])
    purge_days = int(cfg["purge_days"])

    first_test_start = _add_years(min_d, cfg["pretrain_years"])
    if first_test_start > max_d:
        return []

    folds = []
    fold_id = 0
    test_start = first_test_start
    while test_start <= max_d:
        test_end = min(_add_months(test_start, test_months), max_d)
        if cfg["window"] == "expanding":
            train_start = min_d
        else:  # rolling
            train_start = _add_years(test_start, -cfg["rolling_years"])
        train_end = test_start

        tr, te = purge_split(dates, train_end, test_start, test_end, purge_days)
        if len(tr) == 0 or len(te) == 0:
            break

        folds.append(WFFold(
            fold_id=fold_id,
            train_start=tr.min(),
            train_end=tr.max(),
            test_start=te.min(),
            test_end=te.max(),
        ))
        fold_id += 1
        test_start = _add_months(test_start, retrain_months)

    return folds