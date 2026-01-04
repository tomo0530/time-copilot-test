from __future__ import annotations

from pathlib import Path

import pandas as pd

from project_module.config import (
    DatasetConfig,
    ForecastingConfig,
    HoldoutConfig,
    PreprocessingConfig,
)
from project_module.data_preprocessing.base import BasePreprocessor, PreparedDataset
from project_module.data_preprocessing.common import (
    build_unique_id,
    ensure_columns,
    ensure_continuous_dates,
    ensure_datetime_column,
    time_series_holdout_split,
)


class StoreItemPreprocessor(BasePreprocessor):
    """Preprocessor for the Store Item Demand Forecasting dataset."""

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
        Load and preprocess the Store Item dataset.

        Returns
        -------
        PreparedDataset
            Prepared training and test datasets.
        """
        dataset_path = Path(self._dataset_config.path)
        if not dataset_path.is_absolute():
            dataset_path = self._dataset_dir / dataset_path
        train_file = self._dataset_config.train_file or "train.csv"
        df = pd.read_csv(filepath_or_buffer=dataset_path / train_file)

        date_column = self._preprocessing.date_column or "date"
        target_column = self._preprocessing.target_column or "sales"

        ensure_columns(df=df, columns=[date_column, target_column, "store", "item"])
        df = ensure_datetime_column(df=df, column=date_column)
        df["unique_id"] = build_unique_id(
            df=df,
            unique_id_format=self._preprocessing.unique_id_format,
            fallback_column="store",
        )
        df["unique_id"] = df["unique_id"].astype(dtype=str)
        df = ensure_continuous_dates(
            df=df,
            date_column=date_column,
            group_column="unique_id",
            freq=self._preprocessing.freq,
        )
        df["store"] = df.groupby(by="unique_id")["store"].ffill().bfill()
        df["item"] = df.groupby(by="unique_id")["item"].ffill().bfill()
        df[target_column] = df[target_column].fillna(value=0.0).astype(dtype=float)

        train_df, test_df = time_series_holdout_split(
            df=df,
            date_column=date_column,
            horizon=self._forecasting.horizon,
            freq=self._preprocessing.freq,
            holdout=self._holdout,
        )
        train_df = train_df.rename(columns={date_column: "ds", target_column: "y"})
        test_df = test_df.rename(columns={date_column: "ds", target_column: "y"})
        train_df = train_df.sort_values(by=["unique_id", "ds"])
        test_df = test_df.sort_values(by=["unique_id", "ds"])

        metadata = df[["unique_id", "store", "item"]].drop_duplicates()
        return PreparedDataset(
            train_df=train_df[["unique_id", "ds", "y"]],
            test_df=test_df[["unique_id", "ds", "y"]],
            freq=self._preprocessing.freq,
            horizon=self._forecasting.horizon,
            metadata=metadata,
        )
