from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from project_module.data_preprocessing.base import M5DataBundle


@dataclass(frozen=True)
class M5LevelSpec:
    """Specification for an M5 aggregation level."""

    name: str
    group_cols: list[str]
    series_count: int


@dataclass(frozen=True)
class M5WrmsseResult:
    """Result for WRMSSE evaluation."""

    wrmsse: float
    level_scores: dict[str, float]


def get_level_specs() -> list[M5LevelSpec]:
    """
    Return the standard 12 aggregation levels for M5.

    Returns
    -------
    list[M5LevelSpec]
        Level specifications.
    """
    return [
        M5LevelSpec(name="all", group_cols=[], series_count=1),
        M5LevelSpec(name="state", group_cols=["state_id"], series_count=3),
        M5LevelSpec(name="store", group_cols=["store_id"], series_count=10),
        M5LevelSpec(name="category", group_cols=["cat_id"], series_count=3),
        M5LevelSpec(name="department", group_cols=["dept_id"], series_count=7),
        M5LevelSpec(name="state_category", group_cols=["state_id", "cat_id"], series_count=9),
        M5LevelSpec(name="state_department", group_cols=["state_id", "dept_id"], series_count=21),
        M5LevelSpec(name="store_category", group_cols=["store_id", "cat_id"], series_count=30),
        M5LevelSpec(name="store_department", group_cols=["store_id", "dept_id"], series_count=70),
        M5LevelSpec(name="item", group_cols=["item_id"], series_count=3049),
        M5LevelSpec(name="state_item", group_cols=["state_id", "item_id"], series_count=30490),
        M5LevelSpec(name="store_item", group_cols=["store_id", "item_id"], series_count=30490),
    ]


def get_level_spec(level: int) -> M5LevelSpec:
    """
    Return the level specification by numeric level.

    Parameters
    ----------
    level : int
        M5 level index (1-12).

    Returns
    -------
    M5LevelSpec
        Level specification.
    """
    specs = get_level_specs()
    if level < 1 or level > len(specs):
        raise ValueError(f"Unsupported M5 level: {level}")
    return specs[level - 1]


def get_d_columns(start_day: int, end_day: int) -> list[str]:
    """
    Generate M5 day column names.

    Parameters
    ----------
    start_day : int
        Starting day index.
    end_day : int
        Ending day index.

    Returns
    -------
    list[str]
        Day column names.
    """
    return [f"d_{day}" for day in range(start_day, end_day + 1)]


def compute_scale(values: npt.NDArray[np.float64]) -> float:
    """
    Compute RMSSE scale for a time series.

    Parameters
    ----------
    values : numpy.ndarray
        Time series values.

    Returns
    -------
    float
        Scale value.
    """
    non_zero_indices = np.flatnonzero(a=values)
    if non_zero_indices.size <= 1:
        return 0.0
    start_idx = non_zero_indices[0]
    trimmed = values[start_idx:]
    diffs = np.diff(a=trimmed)
    if diffs.size == 0:
        return 0.0
    return float(np.mean(np.square(diffs)))


def aggregate_series(
    values: npt.NDArray[np.float64],
    metadata: pd.DataFrame,
    group_cols: list[str],
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """
    Aggregate series by the specified grouping columns.

    Parameters
    ----------
    values : numpy.ndarray
        Base-level values with shape (n_series, horizon).
    metadata : pandas.DataFrame
        Metadata aligned with base series.
    group_cols : list[str]
        Grouping columns.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Aggregated values and group keys.
    """
    if not group_cols:
        return values.sum(axis=0, keepdims=True), ["all"]
    grouping = (
        metadata[group_cols]
        .astype(dtype=str)
        .agg(
            func="_".join,
            axis=1,
        )
    )
    unique_groups = grouping.drop_duplicates().tolist()
    aggregated = np.zeros(
        shape=(len(unique_groups), values.shape[1]),
        dtype=float,
    )
    for index, group in enumerate(unique_groups):
        mask = grouping == group
        aggregated[index] = values[mask.to_numpy(dtype=bool)].sum(axis=0)
    return aggregated, unique_groups


def compute_bottom_weights(bundle: M5DataBundle) -> pd.Series:
    """
    Compute bottom-level weights based on sales and prices.

    Parameters
    ----------
    bundle : M5DataBundle
        M5 dataset bundle.

    Returns
    -------
    pandas.Series
        Weight per bottom-level series indexed by id.
    """
    sales_df = bundle.sales_df
    calendar_df = bundle.calendar_df
    prices_df = bundle.prices_df
    d_columns = get_d_columns(bundle.train_end_day - 27, bundle.train_end_day)
    base = sales_df[["id", "item_id", "store_id"] + d_columns]
    long_df = base.melt(
        id_vars=["id", "item_id", "store_id"],
        value_vars=d_columns,
        var_name="d",
        value_name="sales",
    )
    long_df = long_df.merge(
        right=calendar_df[["d", "wm_yr_wk"]],
        on="d",
        how="left",
    )
    long_df = long_df.merge(
        right=prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    long_df["sell_price"] = long_df["sell_price"].fillna(value=0.0)
    long_df["revenue"] = long_df["sales"] * long_df["sell_price"]
    weights = long_df.groupby(by="id")["revenue"].sum()
    total = weights.sum()
    if total == 0:
        return weights
    normalized = weights / float(total)
    return cast(npt.NDArray[np.float64], normalized)


def compute_wrmsse(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    bundle: M5DataBundle,
) -> M5WrmsseResult:
    """
    Compute WRMSSE for M5 predictions.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Predictions with columns ["unique_id", "ds", "yhat"].
    actuals : pandas.DataFrame
        Actual values with columns ["unique_id", "ds", "y"].
    bundle : M5DataBundle
        M5 dataset bundle.

    Returns
    -------
    M5WrmsseResult
        WRMSSE result with per-level scores.
    """
    sales_df = bundle.sales_df
    metadata = sales_df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].rename(
        columns={"id": "unique_id"}
    )
    series_ids = metadata["unique_id"].tolist()

    train_d_columns = get_d_columns(1, bundle.train_end_day)
    test_d_columns = get_d_columns(bundle.test_start_day, bundle.test_end_day)

    train_values = sales_df[train_d_columns].to_numpy(dtype=float)

    test_dates = bundle.calendar_df[
        bundle.calendar_df["d"].isin(values=test_d_columns)
    ].sort_values(by="d")["date"]
    test_dates = pd.to_datetime(arg=test_dates).tolist()
    actuals = actuals.copy()
    actuals["ds"] = pd.to_datetime(arg=actuals["ds"])
    actuals_pivot = actuals.pivot_table(
        index="unique_id",
        columns="ds",
        values="y",
    )
    actuals_aligned = actuals_pivot.reindex(index=series_ids, columns=test_dates)
    if actuals_aligned.isnull().any().any():
        raise ValueError("Actuals are missing values for the M5 horizon.")
    test_values = actuals_aligned.to_numpy(dtype=float)

    pred_pivot = predictions.pivot_table(
        index="unique_id",
        columns="ds",
        values="yhat",
    )
    pred_aligned = pred_pivot.reindex(index=series_ids, columns=test_dates)
    if pred_aligned.isnull().any().any():
        raise ValueError("Predictions are missing values for the M5 horizon.")
    pred_values = pred_aligned.to_numpy(dtype=float)

    base_weights = compute_bottom_weights(bundle=bundle)
    base_weights = base_weights.reindex(index=series_ids).fillna(value=0.0)

    level_scores: dict[str, float] = {}
    level_specs = get_level_specs()
    for spec in level_specs:
        agg_train, _ = aggregate_series(
            values=train_values,
            metadata=metadata,
            group_cols=spec.group_cols,
        )
        agg_true, _ = aggregate_series(
            values=test_values,
            metadata=metadata,
            group_cols=spec.group_cols,
        )
        agg_pred, group_keys = aggregate_series(
            values=pred_values,
            metadata=metadata,
            group_cols=spec.group_cols,
        )
        weights = aggregate_weights(
            base_weights=base_weights,
            metadata=metadata,
            group_cols=spec.group_cols,
            group_keys=group_keys,
        )
        scales = np.array([compute_scale(series) for series in agg_train])
        rmse = np.sqrt(np.mean(np.square(agg_pred - agg_true), axis=1))
        valid = scales > 0
        rmsse = np.full_like(a=rmse, fill_value=np.nan, dtype=float)
        rmsse[valid] = rmse[valid] / np.sqrt(scales[valid])
        if weights.sum() == 0:
            level_scores[spec.name] = float("nan")
            continue
        normalized_weights = normalize_weights(weights=weights, valid_mask=valid)
        level_scores[spec.name] = float(np.nansum(a=normalized_weights * rmsse))

    wrmsse = float(np.nanmean(a=list(level_scores.values())))
    return M5WrmsseResult(wrmsse=wrmsse, level_scores=level_scores)


def compute_wrmsse_for_level(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    bundle: M5DataBundle,
    level: int,
) -> M5WrmsseResult:
    """
    Compute WRMSSE for a single M5 aggregation level.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Predictions with columns ["unique_id", "ds", "yhat"].
    actuals : pandas.DataFrame
        Actual values with columns ["unique_id", "ds", "y"].
    bundle : M5DataBundle
        M5 dataset bundle.
    level : int
        M5 aggregation level (1-12).

    Returns
    -------
    M5WrmsseResult
        WRMSSE result for the requested level.
    """
    spec = get_level_spec(level=level)
    sales_df = bundle.sales_df
    metadata = sales_df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].rename(
        columns={"id": "unique_id"}
    )
    series_ids = metadata["unique_id"].tolist()

    train_d_columns = get_d_columns(1, bundle.train_end_day)
    test_d_columns = get_d_columns(bundle.test_start_day, bundle.test_end_day)

    train_values = sales_df[train_d_columns].to_numpy(dtype=float)
    test_dates = bundle.calendar_df[
        bundle.calendar_df["d"].isin(values=test_d_columns)
    ].sort_values(by="d")["date"]
    test_dates = pd.to_datetime(arg=test_dates).tolist()

    agg_train, group_keys = aggregate_series(
        values=train_values,
        metadata=metadata,
        group_cols=spec.group_cols,
    )

    actuals = actuals.copy()
    actuals["ds"] = pd.to_datetime(arg=actuals["ds"])
    actuals_pivot = actuals.pivot_table(
        index="unique_id",
        columns="ds",
        values="y",
    )
    actuals_aligned = actuals_pivot.reindex(index=group_keys, columns=test_dates)
    if actuals_aligned.isnull().any().any():
        raise ValueError("Actuals are missing values for the M5 horizon.")
    test_values = actuals_aligned.to_numpy(dtype=float)

    pred_pivot = predictions.pivot_table(
        index="unique_id",
        columns="ds",
        values="yhat",
    )
    pred_aligned = pred_pivot.reindex(index=group_keys, columns=test_dates)
    if pred_aligned.isnull().any().any():
        raise ValueError("Predictions are missing values for the M5 horizon.")
    pred_values = pred_aligned.to_numpy(dtype=float)

    base_weights = compute_bottom_weights(bundle=bundle)
    base_weights = base_weights.reindex(index=series_ids).fillna(value=0.0)
    weights = aggregate_weights(
        base_weights=base_weights,
        metadata=metadata,
        group_cols=spec.group_cols,
        group_keys=group_keys,
    )
    scales = np.array([compute_scale(series) for series in agg_train])
    rmse = np.sqrt(np.mean(np.square(pred_values - test_values), axis=1))
    valid = scales > 0
    rmsse = np.full_like(a=rmse, fill_value=np.nan, dtype=float)
    rmsse[valid] = rmse[valid] / np.sqrt(scales[valid])
    if weights.sum() == 0:
        return M5WrmsseResult(wrmsse=float("nan"), level_scores={spec.name: float("nan")})
    normalized_weights = normalize_weights(weights=weights, valid_mask=valid)
    score = float(np.nansum(a=normalized_weights * rmsse))
    return M5WrmsseResult(wrmsse=score, level_scores={spec.name: score})


def aggregate_weights(
    base_weights: pd.Series,
    metadata: pd.DataFrame,
    group_cols: list[str],
    group_keys: list[str],
) -> npt.NDArray[np.float64]:
    """
    Aggregate bottom-level weights for a hierarchy level.

    Parameters
    ----------
    base_weights : pandas.Series
        Base weights indexed by unique_id.
    metadata : pandas.DataFrame
        Metadata aligned with base series.
    group_cols : list[str]
        Grouping columns.
    group_keys : list[str]
        Group keys.

    Returns
    -------
    numpy.ndarray
        Aggregated weights.
    """
    if not group_cols:
        return np.array([base_weights.sum()], dtype=float)
    grouping = (
        metadata[group_cols]
        .astype(dtype=str)
        .agg(
            func="_".join,
            axis=1,
        )
    )
    weights = np.zeros(len(group_keys), dtype=float)
    for index, key in enumerate(group_keys):
        mask = grouping == key
        weights[index] = base_weights[mask.to_numpy(dtype=bool)].sum()
    return weights


def normalize_weights(
    weights: npt.NDArray[np.float64],
    valid_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """
    Normalize weights, skipping invalid entries.

    Parameters
    ----------
    weights : numpy.ndarray
        Raw weights.
    valid_mask : numpy.ndarray
        Mask for valid values.

    Returns
    -------
    numpy.ndarray
        Normalized weights.
    """
    weights = weights.astype(dtype=float)
    weights[~valid_mask] = 0.0
    total = weights.sum()
    if total == 0:
        return weights
    normalized = weights / float(total)
    return cast(npt.NDArray[np.float64], normalized)
