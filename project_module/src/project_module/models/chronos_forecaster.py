from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from chronos.chronos2 import Chronos2Pipeline

from project_module.models.base import ForecastModel, ModelResult

LOGGER = logging.getLogger("custom_logger")


@dataclass(frozen=True)
class ChronosConfig:
    """Configuration for the Chronos forecaster."""

    repo_id: str
    context_length: int
    batch_size: int
    quantile_levels: list[float]


def build_context_tensor(
    train_df: pd.DataFrame,
    context_length: int,
) -> tuple[torch.Tensor, list[object]]:
    """
    Build a context tensor for Chronos inference.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data.
    context_length : int
        Context window length.

    Returns
    -------
    tuple[torch.Tensor, list[str]]
        Context tensor (n_series, 1, history_length) and corresponding unique_ids.
    """
    sequences: list[npt.NDArray[np.float32]] = []
    unique_ids: list[object] = []
    for unique_id, series_df in train_df.groupby(by="unique_id"):
        series_values = series_df.sort_values(by="ds")["y"].to_numpy(dtype=float)
        if len(series_values) >= context_length:
            context = series_values[-context_length:]
        else:
            padding = np.zeros(
                shape=context_length - len(series_values),
                dtype=float,
            )
            context = np.concatenate((padding, series_values))
        sequences.append(context.astype(dtype=np.float32))
        unique_ids.append(unique_id)
    tensor = torch.tensor(data=np.stack(arrays=sequences, axis=0))
    tensor = tensor.unsqueeze(dim=1)
    return tensor, unique_ids


def _build_chronos_pipeline_compat(
    repo_id: str,
    device_map: str,
) -> Chronos2Pipeline:
    return Chronos2Pipeline.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        device_map=device_map,
    )


class ChronosForecaster(ForecastModel):
    """Chronos-2 forecaster using the Chronos pipeline."""

    def __init__(self, config: ChronosConfig | None = None) -> None:
        if config is None:
            config = ChronosConfig(
                repo_id="amazon/chronos-2",
                context_length=256,
                batch_size=256,
                quantile_levels=[0.5],
            )
        self._config = config

    @property
    def name(self) -> str:
        return "Chronos-2"

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        freq: str,
    ) -> ModelResult:
        """
        Run zero-shot Chronos-2 forecasting.

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
        if torch.cuda.is_available():
            device_map = "cuda"
        else:
            device_map = "cpu"
        LOGGER.info(
            msg=(
                f"Chronos-2 loading pipeline: repo_id={self._config.repo_id}, device={device_map}."
            )
        )
        load_start = time.perf_counter()
        pipeline = Chronos2Pipeline.from_pretrained(
            pretrained_model_name_or_path=self._config.repo_id,
            device_map=device_map,
        )
        LOGGER.info(msg=f"Chronos-2 pipeline loaded in {time.perf_counter() - load_start:.2f}s.")
        context_tensor, unique_ids = build_context_tensor(
            train_df=train_df,
            context_length=self._config.context_length,
        )
        LOGGER.info(
            msg=(
                "Chronos-2 context ready: "
                f"series={len(unique_ids)}, context_length={self._config.context_length}."
            )
        )
        LOGGER.info(
            msg=(
                "Chronos-2 prediction start: "
                f"horizon={horizon}, quantiles={self._config.quantile_levels}."
            )
        )
        predict_start = time.perf_counter()
        quantiles, _ = pipeline.predict_quantiles(
            inputs=context_tensor,
            prediction_length=horizon,
            quantile_levels=self._config.quantile_levels,
        )
        LOGGER.info(
            msg=f"Chronos-2 prediction complete in {time.perf_counter() - predict_start:.2f}s."
        )
        if isinstance(quantiles, list):
            quantiles_tensor = torch.stack(quantiles)
        else:
            quantiles_tensor = torch.as_tensor(quantiles)
        if quantiles_tensor.dim() == 3:
            quantiles_tensor = quantiles_tensor.unsqueeze(dim=1)
        if quantiles_tensor.dim() != 4:
            raise ValueError(
                "Chronos-2 quantiles must be 4D: (n_series, n_variates, horizon, n_quantiles)."
            )
        if 0.5 in self._config.quantile_levels:
            quantile_index = self._config.quantile_levels.index(0.5)
        else:
            quantile_index = len(self._config.quantile_levels) // 2
        median = quantiles_tensor[:, 0, :, quantile_index].detach().cpu().numpy()
        predictions: list[pd.DataFrame] = []
        total_series = len(unique_ids)
        log_interval = max(1, total_series // 10)
        LOGGER.info(
            msg=(f"Chronos-2 assembling forecasts: series={total_series}, horizon={horizon}.")
        )
        for index, unique_id in enumerate(unique_ids):
            series_dates = pd.to_datetime(
                arg=train_df.loc[train_df["unique_id"] == unique_id, "ds"]
            )
            last_date = series_dates.max()
            future_dates = pd.date_range(
                start=last_date + pd.tseries.frequencies.to_offset(freq=freq),
                periods=horizon,
                freq=freq,
            )
            series_pred = pd.DataFrame(
                data={
                    "unique_id": str(unique_id),
                    "ds": future_dates,
                    "yhat": median[index],
                }
            )
            predictions.append(series_pred)
            if (index + 1) % log_interval == 0 or (index + 1) == total_series:
                LOGGER.info(
                    msg=(
                        f"Chronos-2 forecast assembly progress: {index + 1}/{total_series} series."
                    )
                )
        pred_df = pd.concat(objs=predictions, ignore_index=True)
        elapsed = time.perf_counter() - start_time
        LOGGER.info(msg=f"Chronos-2 fit_predict complete in {elapsed:.2f}s.")
        return ModelResult(
            model_name=self.name,
            predictions=pred_df,
            train_time_sec=elapsed,
        )
