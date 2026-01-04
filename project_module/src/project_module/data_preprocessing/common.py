from __future__ import annotations

from functools import partial
from typing import Iterable

import pandas as pd

from project_module.config import HoldoutConfig


def ensure_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Ensure that a column is converted to pandas datetime.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    column : str
        Column name to convert.

    Returns
    -------
    pandas.DataFrame
        DataFrame with converted datetime column.
    """
    df = df.copy()
    df[column] = pd.to_datetime(arg=df[column])
    return df


def format_unique_id(row: pd.Series, template: str) -> str:
    """
    Format a unique identifier using a template and a dataframe row.

    Parameters
    ----------
    row : pandas.Series
        Row values.
    template : str
        Format string using column names, e.g. "{store}_{item}".

    Returns
    -------
    str
        Formatted identifier.
    """
    return template.format(**row.to_dict())


def build_unique_id(
    df: pd.DataFrame,
    unique_id_format: str | None,
    fallback_column: str,
) -> pd.Series:
    """
    Build a unique identifier column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    unique_id_format : str | None
        Format template for unique identifiers.
    fallback_column : str
        Column to use if no template is provided.

    Returns
    -------
    pandas.Series
        Unique identifier series.
    """
    if unique_id_format:
        formatter = partial(format_unique_id, template=unique_id_format)
        return df.apply(func=formatter, axis=1)
    return df[fallback_column].astype(dtype=str)


def ensure_continuous_dates(
    df: pd.DataFrame,
    date_column: str,
    group_column: str,
    freq: str,
) -> pd.DataFrame:
    """
    Ensure each series has a continuous date range.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    date_column : str
        Date column name.
    group_column : str
        Grouping column name for series identifiers.
    freq : str
        Pandas frequency string.

    Returns
    -------
    pandas.DataFrame
        DataFrame with continuous dates per series.
    """
    frames: list[pd.DataFrame] = []
    for unique_id, group in df.groupby(by=group_column):
        group = group.sort_values(by=date_column)
        date_range = pd.date_range(
            start=group[date_column].min(),
            end=group[date_column].max(),
            freq=freq,
        )
        reindexed = (
            group.set_index(date_column)
            .reindex(index=date_range)
            .rename_axis(date_column)
            .reset_index()
        )
        reindexed[group_column] = unique_id
        frames.append(reindexed)
    return pd.concat(objs=frames, ignore_index=True)


def _resolve_holdout_dates(
    df: pd.DataFrame,
    date_column: str,
    horizon: int,
    freq: str,
    holdout: HoldoutConfig,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    if holdout.train_end and holdout.test_start and holdout.test_end:
        train_end = pd.Timestamp(ts=holdout.train_end)
        test_start = pd.Timestamp(ts=holdout.test_start)
        test_end = pd.Timestamp(ts=holdout.test_end)
        return train_end, test_start, test_end
    if holdout.train_end and holdout.test_end:
        train_end = pd.Timestamp(ts=holdout.train_end)
        test_end = pd.Timestamp(ts=holdout.test_end)
        test_start = pd.date_range(
            end=test_end,
            periods=horizon,
            freq=freq,
        )[0]
        return train_end, test_start, test_end
    max_date = df[date_column].max()
    if max_date is pd.NaT:
        raise ValueError("Date column contains no valid timestamps.")
    test_dates = pd.date_range(end=max_date, periods=horizon, freq=freq)
    test_start = test_dates[0]
    train_end = test_start - pd.tseries.frequencies.to_offset(freq=freq)
    return train_end, test_start, test_dates[-1]


def time_series_holdout_split(
    df: pd.DataFrame,
    date_column: str,
    horizon: int,
    freq: str,
    holdout: HoldoutConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train/test sets using a time-series holdout.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    date_column : str
        Date column name.
    horizon : int
        Forecast horizon.
    freq : str
        Pandas frequency string.
    holdout : HoldoutConfig
        Holdout configuration.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Train and test splits.
    """
    if holdout.strategy != "time_series_split":
        raise ValueError(f"Unsupported holdout strategy: {holdout.strategy}")
    train_end, test_start, test_end = _resolve_holdout_dates(
        df=df,
        date_column=date_column,
        horizon=horizon,
        freq=freq,
        holdout=holdout,
    )
    train_mask = df[date_column] <= train_end
    test_mask = (df[date_column] >= test_start) & (df[date_column] <= test_end)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """
    Validate that required columns are present.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    columns : Iterable[str]
        Required column names.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
