from __future__ import annotations

import pandas as pd
import pytest

from data.storage import SqliteStorage, StorageConfig


@pytest.fixture()
def st(tmp_path):
    return SqliteStorage(StorageConfig(db_path=tmp_path / "t.sqlite",
                                       artifacts_dir=tmp_path))


def _df(dates, closes):
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [100] * len(closes),
            "value": [1000] * len(closes),
        },
        index=pd.to_datetime(dates),
    )


def test_save_and_load_roundtrip(st):
    st.save_prices("X", _df(["2024-01-01", "2024-01-02"], [10, 11]))
    df = st.load("X")
    assert len(df) == 2
    assert df["close"].tolist() == [10, 11]


def test_dedup_on_repeat(st):
    df = _df(["2024-01-01", "2024-01-02"], [10, 11])
    st.save_prices("X", df)
    st.save_prices("X", df)
    assert len(st.load("X")) == 2


def test_last_date(st):
    assert st.last_date("X") is None
    st.save_prices("X", _df(["2024-01-01", "2024-01-02"], [10, 11]))
    assert st.last_date("X") == "2024-01-02"


def test_different_start_dates_no_intersection(st):
    st.save_prices("A", _df(["2020-01-01"], [1]))
    st.save_prices("B", _df(["2023-01-01"], [2]))
    assert st.load("A")["date"].min() < st.load("B")["date"].min()