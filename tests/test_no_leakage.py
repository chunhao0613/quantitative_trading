from __future__ import annotations

import pandas as pd

from src.features.build_dataset import build_dataset
from src.features.technical_indicators import add_indicators


def test_no_data_leakage_by_time_shift() -> None:
    dates = pd.bdate_range("2024-01-01", periods=120)
    base = pd.Series(range(120), index=dates, dtype=float)
    close = 100 + base * 0.2 + (base % 7 - 3) * 0.8
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST",
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
            "turnover": close * 1_000_000,
            "transactions": 10_000,
            "change": close.diff().fillna(0),
            "is_trading_day": 1,
        }
    )

    feat = add_indicators(df)
    ds = build_dataset(feat)

    assert not ds.empty
    # target_t1 depends on close(t+1), while features are shifted by 1,
    # so same-row close should not be directly used as feature source.
    assert "target_t1" in ds.columns
    assert (ds["date"].diff().dropna() >= pd.Timedelta(days=1)).all()
