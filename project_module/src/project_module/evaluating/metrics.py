from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def smape(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    float
        SMAPE value.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    mask = denominator != 0
    smape_values = np.zeros_like(diff, dtype=float)
    smape_values[mask] = diff[mask] / denominator[mask]
    return float(np.mean(smape_values))


def rmspe(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    """
    Compute Root Mean Squared Percentage Error (RMSPE).

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSPE value.
    """
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    percentage_errors = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(np.square(percentage_errors))))


def evaluate_predictions(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    metric_name: str,
) -> float:
    """
    Evaluate predictions against actuals.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Predictions with columns ["unique_id", "ds", "yhat"].
    actuals : pandas.DataFrame
        Actual values with columns ["unique_id", "ds", "y"].
    metric_name : str
        Metric name ("smape" or "rmspe").

    Returns
    -------
    float
        Metric value.
    """
    merged = actuals.merge(
        right=predictions,
        on=["unique_id", "ds"],
        how="inner",
    )
    y_true = merged["y"].to_numpy(dtype=float)
    y_pred = merged["yhat"].to_numpy(dtype=float)
    if metric_name == "smape":
        return smape(y_true=y_true, y_pred=y_pred)
    if metric_name == "rmspe":
        return rmspe(y_true=y_true, y_pred=y_pred)
    raise ValueError(f"Unsupported metric: {metric_name}")
