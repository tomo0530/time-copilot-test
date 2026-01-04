from __future__ import annotations

import pandas as pd

from project_module.config import HoldoutConfig
from project_module.data_preprocessing.common import time_series_holdout_split


def test_holdout_split_uses_last_horizon() -> None:
    dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(data={"unique_id": ["A"] * 5, "ds": dates, "y": [1, 2, 3, 4, 5]})
    holdout = HoldoutConfig(
        strategy="time_series_split",
        train_end=None,
        test_start=None,
        test_end=None,
        train_end_day=None,
        test_start_day=None,
        test_end_day=None,
    )
    train_df, test_df = time_series_holdout_split(
        df=df,
        date_column="ds",
        horizon=2,
        freq="D",
        holdout=holdout,
    )
    assert train_df["ds"].max() == pd.Timestamp("2020-01-03")
    assert test_df["ds"].min() == pd.Timestamp("2020-01-04")
    assert len(test_df) == 2
