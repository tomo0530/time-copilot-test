from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from project_module.config import (
    DatasetConfig,
    ForecastingConfig,
    HoldoutConfig,
    PreprocessingConfig,
)


@dataclass(frozen=True)
class M5DataBundle:
    """Bundle of M5 raw artifacts for WRMSSE computation."""

    sales_df: pd.DataFrame
    calendar_df: pd.DataFrame
    prices_df: pd.DataFrame
    id_columns: list[str]
    train_end_day: int
    test_start_day: int
    test_end_day: int


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared dataset with train/test splits and metadata."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    freq: str
    horizon: int
    metadata: pd.DataFrame
    m5_bundle: M5DataBundle | None = None


class BasePreprocessor(ABC):
    """Base class for dataset preprocessors."""

    def __init__(
        self,
        dataset_dir: Path,
        dataset_config: DatasetConfig,
        preprocessing: PreprocessingConfig,
        holdout: HoldoutConfig,
        forecasting: ForecastingConfig,
    ) -> None:
        """
        Initialize the preprocessor.

        Parameters
        ----------
        dataset_dir : Path
            Base dataset directory.
        dataset_config : DatasetConfig
            Dataset file configuration.
        preprocessing : PreprocessingConfig
            Preprocessing configuration.
        holdout : HoldoutConfig
            Holdout split configuration.
        forecasting : ForecastingConfig
            Forecasting configuration.
        """
        self._dataset_dir = dataset_dir
        self._dataset_config = dataset_config
        self._preprocessing = preprocessing
        self._holdout = holdout
        self._forecasting = forecasting

    @abstractmethod
    def prepare(self) -> PreparedDataset:
        """
        Prepare the dataset for modeling.

        Returns
        -------
        PreparedDataset
            Prepared training/test data and metadata.
        """
        raise NotImplementedError
