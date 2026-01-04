from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMRegressor

from project_module.models.base import ForecastModel, ModelResult

LOGGER = logging.getLogger("custom_logger")

DATE_FEATURE_COLUMNS = [
    "dayofweek",
    "weekofyear",
    "month",
    "day",
    "dayofyear",
]


@dataclass(frozen=True)
class LightGBMConfig:
    """Configuration for the LightGBM forecaster."""

    lags: list[int]
    num_leaves: int
    n_estimators: int
    learning_rate: float
    random_state: int


def add_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Add standard calendar features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    date_column : str
        Date column name.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added date features.
    """
    df = df.copy()
    df["dayofweek"] = df[date_column].dt.dayofweek
    df["weekofyear"] = df[date_column].dt.isocalendar().week.astype(dtype=int)
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["dayofyear"] = df[date_column].dt.dayofyear
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Add lag features to the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with columns ["unique_id", "y"].
    lags : list[int]
        Lag values.

    Returns
    -------
    pandas.DataFrame
        DataFrame with lag columns.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(by="unique_id")["y"].shift(lag)
    return df


def encode_unique_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Encode unique_id column into integer codes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, int]]
        DataFrame with encoded unique_id and the mapping.
    """
    df = df.copy()
    df["unique_id"] = df["unique_id"].astype(dtype="category")
    df["unique_id_code"] = df["unique_id"].cat.codes
    mapping = {
        str(category): int(code) for code, category in enumerate(df["unique_id"].cat.categories)
    }
    return df, mapping


def build_training_frame(
    train_df: pd.DataFrame,
    lags: list[int],
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """
    Build the training matrix for LightGBM.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data.
    lags : list[int]
        Lag values.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, dict[str, int]]
        Feature matrix, target series, and id mapping.
    """
    df = train_df.copy()
    df["ds"] = pd.to_datetime(arg=df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    df = add_date_features(df=df, date_column="ds")
    df = add_lag_features(df=df, lags=lags)
    df, mapping = encode_unique_ids(df=df)
    df = df.dropna()
    feature_columns = ["unique_id_code"] + [f"lag_{lag}" for lag in lags] + DATE_FEATURE_COLUMNS
    return df[feature_columns], df["y"], mapping


def build_future_dates(last_date: pd.Timestamp, horizon: int, freq: str) -> pd.DatetimeIndex:
    """
    Generate future dates for forecasting.

    Parameters
    ----------
    last_date : pandas.Timestamp
        Last date in the training data.
    horizon : int
        Forecast horizon.
    freq : str
        Pandas frequency string.

    Returns
    -------
    pandas.DatetimeIndex
        Future dates.
    """
    return pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq=freq),
        periods=horizon,
        freq=freq,
    )


def build_feature_row(
    history: list[float],
    date_value: pd.Timestamp,
    lags: list[int],
    unique_id_code: int,
) -> npt.NDArray[np.float64]:
    """
    Build a feature row for recursive forecasting.

    Parameters
    ----------
    history : list[float]
        Historical values for the series.
    date_value : pandas.Timestamp
        Current forecast date.
    lags : list[int]
        Lag values.
    unique_id_code : int
        Encoded unique id.

    Returns
    -------
    numpy.ndarray
        Feature row.
    """
    lag_values = [history[-lag] for lag in lags]
    features = [
        unique_id_code,
        *lag_values,
        date_value.dayofweek,
        int(date_value.isocalendar().week),
        date_value.month,
        date_value.day,
        date_value.dayofyear,
    ]
    return np.array(features, dtype=float)


def recursive_forecast_series(
    model: LGBMRegressor,
    series_df: pd.DataFrame,
    horizon: int,
    freq: str,
    lags: list[int],
    unique_id_code: int,
) -> list[float]:
    """
    Forecast a single series recursively.

    Parameters
    ----------
    model : LGBMRegressor
        Trained LightGBM model.
    series_df : pandas.DataFrame
        Training data for the series.
    horizon : int
        Forecast horizon.
    freq : str
        Pandas frequency string.
    lags : list[int]
        Lag values.
    unique_id_code : int
        Encoded unique id.

    Returns
    -------
    list[float]
        Forecast values.
    """
    history = series_df["y"].tolist()
    max_lag = max(lags)
    if len(history) < max_lag:
        padding = [history[0]] * (max_lag - len(history))
        history = padding + history
    last_date = series_df["ds"].max()
    future_dates = build_future_dates(last_date=last_date, horizon=horizon, freq=freq)
    predictions: list[float] = []
    for date_value in future_dates:
        features = build_feature_row(
            history=history,
            date_value=date_value,
            lags=lags,
            unique_id_code=unique_id_code,
        )
        yhat = float(model.predict(X=features.reshape(1, -1))[0])
        predictions.append(yhat)
        history.append(yhat)
    return predictions


class LightGBMForecaster(ForecastModel):
    """LightGBM forecaster using global lag features."""

    def __init__(self, config: LightGBMConfig | None = None) -> None:
        if config is None:
            config = LightGBMConfig(
                lags=[1, 7, 14, 28],
                num_leaves=31,
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
            )
        self._config = config

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        freq: str,
    ) -> ModelResult:
        """
        Fit the LightGBM model and forecast future values.

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
        X, y, mapping = build_training_frame(train_df=train_df, lags=self._config.lags)
        LOGGER.info(
            msg=(
                "LightGBM training start: rows="
                f"{len(X)}, features={X.shape[1]}, lags={self._config.lags}, "
                f"n_estimators={self._config.n_estimators}, "
                f"num_leaves={self._config.num_leaves}, "
                f"learning_rate={self._config.learning_rate}."
            )
        )
        model = LGBMRegressor(
            num_leaves=self._config.num_leaves,
            n_estimators=self._config.n_estimators,
            learning_rate=self._config.learning_rate,
            random_state=self._config.random_state,
        )
        train_start = time.perf_counter()
        model.fit(X=X, y=y)
        train_elapsed = time.perf_counter() - train_start
        LOGGER.info(msg=f"LightGBM training complete in {train_elapsed:.2f}s.")
        predictions = []
        total_series = train_df["unique_id"].nunique()
        log_interval = max(1, total_series // 10)
        LOGGER.info(
            msg=(
                "LightGBM forecasting start: "
                f"series={total_series}, horizon={horizon}, freq={freq}."
            )
        )
        for index, (unique_id, series_df) in enumerate(
            train_df.groupby(by="unique_id"),
            start=1,
        ):
            series_df = series_df.sort_values(by="ds")
            series_preds = recursive_forecast_series(
                model=model,
                series_df=series_df,
                horizon=horizon,
                freq=freq,
                lags=self._config.lags,
                unique_id_code=mapping[str(unique_id)],
            )
            last_date = series_df["ds"].max()
            future_dates = build_future_dates(
                last_date=last_date,
                horizon=horizon,
                freq=freq,
            )
            series_output = pd.DataFrame(
                data={
                    "unique_id": unique_id,
                    "ds": future_dates,
                    "yhat": series_preds,
                }
            )
            predictions.append(series_output)
            if index % log_interval == 0 or index == total_series:
                LOGGER.info(msg=f"LightGBM forecasting progress: {index}/{total_series} series.")
        pred_df = pd.concat(objs=predictions, ignore_index=True)
        elapsed = time.perf_counter() - start_time
        LOGGER.info(msg=f"LightGBM forecasting complete in {elapsed:.2f}s.")
        return ModelResult(
            model_name=self.name,
            predictions=pred_df,
            train_time_sec=elapsed,
        )
