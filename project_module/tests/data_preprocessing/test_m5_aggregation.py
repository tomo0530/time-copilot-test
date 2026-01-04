from __future__ import annotations

import pandas as pd

from project_module.data_preprocessing.m5 import (
    M5_LEVEL_6_GROUP_COLS,
    aggregate_m5_long_df,
)


def test_aggregate_m5_long_df_level6() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    long_df = pd.DataFrame(
        {
            "state_id": ["CA", "CA", "TX", "TX"],
            "cat_id": ["FOODS", "FOODS", "FOODS", "FOODS"],
            "d": ["d_1", "d_2", "d_1", "d_2"],
            "ds": [dates[0], dates[1], dates[0], dates[1]],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    aggregated, metadata = aggregate_m5_long_df(
        long_df=long_df,
        group_cols=M5_LEVEL_6_GROUP_COLS,
    )

    assert set(metadata["unique_id"]) == {"CA_FOODS", "TX_FOODS"}
    assert set(aggregated["unique_id"]) == {"CA_FOODS", "TX_FOODS"}
    assert aggregated.shape[0] == 4
