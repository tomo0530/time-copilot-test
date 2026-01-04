from __future__ import annotations

from pathlib import Path

import pandas as pd

from project_module.config import (
    DatasetConfig,
    ForecastingConfig,
    HoldoutConfig,
    PreprocessingConfig,
)
from project_module.data_preprocessing.base import BasePreprocessor, M5DataBundle, PreparedDataset
from project_module.data_preprocessing.common import ensure_columns


def extract_day_number(value: str) -> int:
    """
    Extract day number from a M5 day column.

    Parameters
    ----------
    value : str
        Day identifier such as "d_1".

    Returns
    -------
    int
        Extracted day number.
    """
    return int(value.split("_")[1])


M5_LEVEL_6_GROUP_COLS = ["state_id", "cat_id"]


def aggregate_m5_long_df(
    long_df: pd.DataFrame,
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate M5 long-form data to the requested hierarchy level.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long-form M5 data with columns including group keys, d, ds, and y.
    group_cols : list[str]
        Columns used to define the aggregation level.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Aggregated long-form data and metadata for the aggregated series.
    """
    ensure_columns(df=long_df, columns=group_cols + ["d", "ds", "y"])
    working = long_df.copy()
    for column in group_cols:
        working[column] = working[column].astype(dtype=str)
    working["unique_id"] = working[group_cols].agg(func="_".join, axis=1)
    aggregated = working.groupby(by=["unique_id", "d", "ds"], as_index=False)["y"].sum()
    metadata = working[group_cols].drop_duplicates().copy()
    metadata["unique_id"] = metadata[group_cols].agg(func="_".join, axis=1)
    metadata = metadata[["unique_id"] + group_cols].drop_duplicates()
    return aggregated, metadata


class M5Preprocessor(BasePreprocessor):
    """Preprocessor for the M5 Forecasting Accuracy dataset."""

    def __init__(
        self,
        dataset_dir: Path,
        dataset_config: DatasetConfig,
        preprocessing: PreprocessingConfig,
        holdout: HoldoutConfig,
        forecasting: ForecastingConfig,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            preprocessing=preprocessing,
            holdout=holdout,
            forecasting=forecasting,
        )

    def prepare(self) -> PreparedDataset:
        """
        Load and preprocess the M5 dataset.

        Returns
        -------
        PreparedDataset
            Prepared training and test datasets with M5 artifacts.
        """
        dataset_path = Path(self._dataset_config.path)
        if not dataset_path.is_absolute():
            dataset_path = self._dataset_dir / dataset_path
        sales_file = self._dataset_config.sales_file or "sales_train_evaluation.csv"
        calendar_file = self._dataset_config.calendar_file or "calendar.csv"
        prices_file = self._dataset_config.prices_file or "sell_prices.csv"

        sales_df = pd.read_csv(filepath_or_buffer=dataset_path / sales_file)
        calendar_df = pd.read_csv(filepath_or_buffer=dataset_path / calendar_file)
        prices_df = pd.read_csv(filepath_or_buffer=dataset_path / prices_file)

        id_columns = [
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ]
        ensure_columns(df=sales_df, columns=id_columns)
        ensure_columns(df=calendar_df, columns=["d", "date", "wm_yr_wk"])
        ensure_columns(df=prices_df, columns=["store_id", "item_id", "wm_yr_wk"])

        calendar_df["date"] = pd.to_datetime(arg=calendar_df["date"])
        d_columns = [column for column in sales_df.columns if column.startswith("d_")]
        if not d_columns:
            raise ValueError("No daily columns (d_*) found in M5 sales data.")

        aggregation_level = self._dataset_config.m5_aggregation_level
        if aggregation_level is not None and aggregation_level != 6:
            raise ValueError(
                f"Unsupported M5 aggregation level: {aggregation_level}. Only level 6 is supported."
            )

        long_df = sales_df.melt(
            id_vars=id_columns,
            value_vars=d_columns,
            var_name="d",
            value_name="y",
        )
        long_df = long_df.merge(
            right=calendar_df[["d", "date"]],
            on="d",
            how="left",
        )
        long_df = long_df.rename(columns={"date": "ds"})
        if aggregation_level == 6:
            long_df, metadata = aggregate_m5_long_df(
                long_df=long_df,
                group_cols=M5_LEVEL_6_GROUP_COLS,
            )
        else:
            long_df = long_df.rename(columns={"id": "unique_id"})
            long_df["unique_id"] = long_df["unique_id"].astype(dtype=str)
            metadata = sales_df[id_columns].rename(columns={"id": "unique_id"})
            metadata["unique_id"] = metadata["unique_id"].astype(dtype=str)

        train_end_day = self._holdout.train_end_day or 1913
        test_start_day = self._holdout.test_start_day or 1914
        test_end_day = self._holdout.test_end_day or 1941

        day_numbers = long_df["d"].apply(func=extract_day_number)
        train_mask = day_numbers <= train_end_day
        test_mask = day_numbers.between(test_start_day, test_end_day)
        train_df = long_df.loc[train_mask, ["unique_id", "ds", "y"]].copy()
        test_df = long_df.loc[test_mask, ["unique_id", "ds", "y"]].copy()
        train_df["y"] = train_df["y"].astype(dtype=float)
        test_df["y"] = test_df["y"].astype(dtype=float)
        train_df = train_df.sort_values(by=["unique_id", "ds"])
        test_df = test_df.sort_values(by=["unique_id", "ds"])

        bundle = M5DataBundle(
            sales_df=sales_df,
            calendar_df=calendar_df,
            prices_df=prices_df,
            id_columns=id_columns,
            train_end_day=train_end_day,
            test_start_day=test_start_day,
            test_end_day=test_end_day,
        )

        return PreparedDataset(
            train_df=train_df,
            test_df=test_df,
            freq=self._preprocessing.freq,
            horizon=self._forecasting.horizon,
            metadata=metadata,
            m5_bundle=bundle,
        )
