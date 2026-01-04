from __future__ import annotations

import numpy as np
import pandas as pd

from project_module.data_preprocessing.base import M5DataBundle
from project_module.evaluating.wrmsse import compute_wrmsse_for_level


def test_compute_wrmsse_for_level_returns_zero_for_perfect_predictions() -> None:
    days = list(range(1, 34))
    d_columns = [f"d_{day}" for day in days]
    dates = pd.date_range(start="2020-01-01", periods=len(days), freq="D")

    sales_rows = []
    for offset, (state_id, item_id, store_id) in enumerate(
        [("CA", "ITEM_1", "STORE_1"), ("TX", "ITEM_2", "STORE_2")]
    ):
        values = np.arange(1, len(days) + 1, dtype=float) + offset
        row = {
            "id": f"{item_id}_{store_id}",
            "item_id": item_id,
            "dept_id": "DEPT_1",
            "cat_id": "FOODS",
            "store_id": store_id,
            "state_id": state_id,
        }
        row.update({column: value for column, value in zip(d_columns, values, strict=True)})
        sales_rows.append(row)
    sales_df = pd.DataFrame(sales_rows)

    calendar_df = pd.DataFrame(
        {
            "d": d_columns,
            "date": dates,
            "wm_yr_wk": [1] * len(days),
        }
    )
    prices_df = pd.DataFrame(
        {
            "store_id": ["STORE_1", "STORE_2"],
            "item_id": ["ITEM_1", "ITEM_2"],
            "wm_yr_wk": [1, 1],
            "sell_price": [1.0, 1.0],
        }
    )

    bundle = M5DataBundle(
        sales_df=sales_df,
        calendar_df=calendar_df,
        prices_df=prices_df,
        id_columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        train_end_day=30,
        test_start_day=31,
        test_end_day=33,
    )

    test_days = ["d_31", "d_32", "d_33"]
    test_dates = calendar_df[calendar_df["d"].isin(test_days)]["date"].tolist()
    actual_rows = []
    for _, row in sales_df.iterrows():
        unique_id = f"{row['state_id']}_{row['cat_id']}"
        for day, ds in zip(test_days, test_dates, strict=True):
            actual_rows.append(
                {
                    "unique_id": unique_id,
                    "ds": ds,
                    "y": float(row[day]),
                }
            )
    actuals = pd.DataFrame(actual_rows)
    predictions = actuals.rename(columns={"y": "yhat"})

    result = compute_wrmsse_for_level(
        predictions=predictions,
        actuals=actuals,
        bundle=bundle,
        level=6,
    )

    assert np.isclose(result.wrmsse, 0.0)
    assert result.level_scores == {"state_category": 0.0}
