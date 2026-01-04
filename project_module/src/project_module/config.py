from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class PathsConfig(BaseModel):
    """Paths used by the experiment runner."""

    dataset_dir: Path
    output_dir: Path
    log_dir: Path

    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_dir", "output_dir", "log_dir", mode="before")
    @classmethod
    def _to_path(cls, value: str | Path) -> Path:
        """Convert string values into Path objects."""
        return Path(value)


class GPUConfig(BaseModel):
    """GPU configuration."""

    auto_detect: bool
    strategy: Literal["hybrid", "single", "multi"]

    model_config = ConfigDict(extra="forbid")


class DatasetConfig(BaseModel):
    """Dataset file configuration."""

    path: str
    data_file: str | None = None
    train_file: str | None = None
    test_file: str | None = None
    sales_file: str | None = None
    calendar_file: str | None = None
    prices_file: str | None = None
    store_file: str | None = None
    m5_aggregation_level: int | None = None

    model_config = ConfigDict(extra="forbid")


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    unique_id_format: str | None = None
    unique_id_column: str | None = None
    date_column: str | None = None
    target_column: str | None = None
    freq: str
    exclude_zero_sales: bool = False

    model_config = ConfigDict(extra="forbid")


class HoldoutConfig(BaseModel):
    """Holdout split configuration."""

    strategy: Literal["time_series_split"]
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    train_end_day: int | None = None
    test_start_day: int | None = None
    test_end_day: int | None = None

    model_config = ConfigDict(extra="forbid")


class ForecastingConfig(BaseModel):
    """Forecasting configuration."""

    horizon: int

    model_config = ConfigDict(extra="forbid")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metric: Literal["smape", "rmspe", "wrmsse"]
    weight_days: int | None = None
    hierarchy_levels: int | None = None

    model_config = ConfigDict(extra="forbid")


class EvaluationLevelConfig(BaseModel):
    """Evaluation level configuration for M5."""

    name: str
    levels: list[int] | Literal["all"]
    series_count: int
    models: list[str] | Literal["all"] = "all"
    exclude_models: list[str] | None = None
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    name: str
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    holdout: HoldoutConfig
    forecasting: ForecastingConfig
    evaluation: EvaluationConfig
    evaluation_levels: list[EvaluationLevelConfig] | None = None

    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    """Top-level configuration for the experiment runner."""

    paths: PathsConfig
    gpu: GPUConfig
    seed: int
    experiment: ExperimentConfig

    model_config = ConfigDict(extra="forbid")
