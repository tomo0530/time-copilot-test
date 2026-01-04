from __future__ import annotations

import time

import pandas as pd
from prophet import Prophet

from project_module.models.base import ForecastModel, ModelResult


class ProphetForecaster(ForecastModel):
    """Prophet forecaster with per-series models."""

    @property
    def name(self) -> str:
        return "Prophet"

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        freq: str,
    ) -> ModelResult:
        """
        Fit Prophet per series and forecast future values.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training data.
        horizon : int
            Forecast horizon.
        freq : str
            Pandas frequency string.

        Returns
        -------
        ModelResult
            Forecasting result.
        """
        start_time = time.perf_counter()
        predictions = []
        for unique_id, series_df in train_df.groupby(by="unique_id"):
            model = Prophet()
            series_df = series_df.copy()
            series_df["ds"] = pd.to_datetime(arg=series_df["ds"])
            prophet_df = series_df[["ds", "y"]]
            model.fit(df=prophet_df)
            future = model.make_future_dataframe(
                periods=horizon,
                freq=freq,
                include_history=False,
            )
            forecast = model.predict(df=future)
            series_output = forecast[["ds", "yhat"]].copy()
            series_output["unique_id"] = str(unique_id)
            predictions.append(series_output.rename(columns={"yhat": "yhat"}))
        pred_df = pd.concat(objs=predictions, ignore_index=True)
        elapsed = time.perf_counter() - start_time
        return ModelResult(
            model_name=self.name,
            predictions=pred_df[["unique_id", "ds", "yhat"]],
            train_time_sec=elapsed,
        )
