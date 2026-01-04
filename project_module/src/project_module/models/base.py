from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelResult:
    """Result of a forecasting model run."""

    model_name: str
    predictions: pd.DataFrame
    train_time_sec: float


class ForecastModel(ABC):
    """Abstract base class for forecasting models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        raise NotImplementedError

    @abstractmethod
    def fit_predict(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        freq: str,
    ) -> ModelResult:
        """
        Fit the model and produce forecasts.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training dataset with columns ["unique_id", "ds", "y"].
        horizon : int
            Forecast horizon.
        freq : str
            Pandas frequency string.

        Returns
        -------
        ModelResult
            Forecasting result.
        """
        raise NotImplementedError
