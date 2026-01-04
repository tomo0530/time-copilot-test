from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from project_module.timecopilot.discovery import discover_timecopilot_models
from project_module.timecopilot.model_selection import (
    ModelSelectionEntry,
    ModelSelectionLog,
)
from project_module.utils import (
    GPUScheduler,
    allocate_gpus_for_model,
    run_with_visible_gpus,
)
from timecopilot import TimeCopilotForecaster


@dataclass(frozen=True)
class TimeCopilotRunResult:
    """Result of TimeCopilot forecasting."""

    predictions: pd.DataFrame
    ensemble: pd.DataFrame
    train_time_sec: float
    selection_log: ModelSelectionLog


def normalize_forecast_output(
    forecast_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Normalize TimeCopilot forecast output into a standard schema.

    Parameters
    ----------
    forecast_df : pandas.DataFrame
        Forecast output.
    model_name : str
        Model name.

    Returns
    -------
    pandas.DataFrame
        Normalized forecast data with columns ["unique_id", "ds", "yhat", "model"].
    """
    if "unique_id" not in forecast_df.columns and "id" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"id": "unique_id"})
    if "ds" not in forecast_df.columns and "date" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"date": "ds"})
    candidate_columns = [
        col for col in forecast_df.columns if col not in {"unique_id", "ds"}
    ]
    if "yhat" in forecast_df.columns:
        yhat_column = "yhat"
    elif model_name in forecast_df.columns:
        yhat_column = model_name
    elif "y_pred" in forecast_df.columns:
        yhat_column = "y_pred"
    elif "TimesFM" in forecast_df.columns:
        yhat_column = "TimesFM"
    elif len(candidate_columns) == 1:
        yhat_column = candidate_columns[0]
    else:
        raise ValueError(f"Unsupported TimeCopilot output columns: {forecast_df.columns}")
    output = forecast_df[["unique_id", "ds", yhat_column]].rename(columns={yhat_column: "yhat"})
    output["unique_id"] = output["unique_id"].astype(dtype=str)
    output["ds"] = pd.to_datetime(arg=output["ds"])
    output["model"] = model_name
    return output


def run_timecopilot_model(
    model: object,
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
) -> pd.DataFrame:
    """
    Run a single TimeCopilot model forecast.

    Parameters
    ----------
    model : object
        TimeCopilot model instance.
    train_df : pandas.DataFrame
        Training data.
    horizon : int
        Forecast horizon.
    freq : str
        Pandas frequency string.

    Returns
    -------
    pandas.DataFrame
        Forecast output.
    """
    forecaster = TimeCopilotForecaster(models=[model])
    return forecaster.forecast(
        df=train_df,
        h=horizon,
        freq=freq,
    )


def compute_median_ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute median ensemble predictions.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Model predictions with columns ["unique_id", "ds", "yhat", "model"].

    Returns
    -------
    pandas.DataFrame
        Median ensemble predictions.
    """
    grouped = predictions.groupby(by=["unique_id", "ds"])["yhat"].median().reset_index()
    grouped["model"] = "TimeCopilotMedianEnsemble"
    return grouped


def run_timecopilot(
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
    dataset_name: str,
    available_gpus: list[int],
    gpu_scheduler: GPUScheduler | None,
    gpu_strategy: Literal["hybrid", "single", "multi"],
    logger: logging.Logger | None = None,
) -> TimeCopilotRunResult:
    """
    Run TimeCopilot forecasting with all available models.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data.
    horizon : int
        Forecast horizon.
    freq : str
        Pandas frequency string.
    dataset_name : str
        Dataset name for logging.
    available_gpus : list[int]
        Available GPU indices.
    gpu_scheduler : GPUScheduler | None
        GPU scheduler instance.
    gpu_strategy : Literal["hybrid", "single", "multi"]
        GPU strategy configuration.
    logger : logging.Logger | None
        Logger instance.

    Returns
    -------
    TimeCopilotRunResult
        TimeCopilot run result.
    """
    start_time = time.perf_counter()
    logger = logger or logging.getLogger(name=__name__)
    models, skipped_entries, failed_entries = discover_timecopilot_models()
    predictions: list[pd.DataFrame] = []
    selected_models: list[str] = []
    for model in models:
        model_name = type(model).__name__
        if not getattr(model, "alias", None):
            try:
                model.alias = model_name
            except Exception:  # noqa: BLE001
                logger.warning(
                    "TimeCopilot model %s has no alias; outputs may be inconsistent.",
                    model_name,
                )
            else:
                logger.warning(
                    "TimeCopilot model %s missing alias; defaulting to class name.",
                    model_name,
                )
        gpu_ids = allocate_gpus_for_model(
            model_name=model_name,
            scheduler=gpu_scheduler,
            available_gpus=available_gpus,
            strategy=gpu_strategy,
        )
        if gpu_ids:
            logger.info(msg=f"Assigned GPUs {gpu_ids} to TimeCopilot {model_name}.")
        else:
            logger.info(msg=f"Using CPU for TimeCopilot {model_name}.")
        try:
            forecast_df = run_with_visible_gpus(
                gpu_ids=gpu_ids,
                run_fn=run_timecopilot_model,
                model=model,
                train_df=train_df,
                horizon=horizon,
                freq=freq,
            )
            predictions.append(
                normalize_forecast_output(
                    forecast_df=forecast_df,
                    model_name=model_name,
                )
            )
            selected_models.append(model_name)
        except Exception as exc:  # noqa: BLE001
            failed_entries.append(
                ModelSelectionEntry(
                    model_name=model_name,
                    status="failed",
                    reason_code="forecast_failed",
                    stage="forecast",
                    error_message=str(exc),
                )
            )
        finally:
            if gpu_scheduler and gpu_ids:
                gpu_scheduler.release(gpu_ids=gpu_ids)
    if predictions:
        pred_df = pd.concat(objs=predictions, ignore_index=True)
    else:
        pred_df = pd.DataFrame(columns=["unique_id", "ds", "yhat", "model"])
    if pred_df.empty:
        ensemble_df = pd.DataFrame(columns=["unique_id", "ds", "yhat", "model"])
        skipped_entries.append(
            ModelSelectionEntry(
                model_name="TimeCopilotMedianEnsemble",
                status="skipped",
                reason_code="no_successful_models",
                stage="ensemble",
                error_message="No successful models available for ensemble.",
            )
        )
    else:
        ensemble_df = compute_median_ensemble(predictions=pred_df)
        if ensemble_df.empty:
            skipped_entries.append(
                ModelSelectionEntry(
                    model_name="TimeCopilotMedianEnsemble",
                    status="skipped",
                    reason_code="empty_predictions",
                    stage="ensemble",
                    error_message="Median ensemble produced no predictions.",
                )
            )
        else:
            selected_models.append("TimeCopilotMedianEnsemble")
    elapsed = time.perf_counter() - start_time
    selection_log = ModelSelectionLog(
        dataset_name=dataset_name,
        selected_models=selected_models,
        skipped_models=skipped_entries,
        failed_models=failed_entries,
    )
    return TimeCopilotRunResult(
        predictions=pred_df,
        ensemble=ensemble_df,
        train_time_sec=elapsed,
        selection_log=selection_log,
    )
