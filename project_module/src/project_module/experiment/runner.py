from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from project_module.config import AppConfig, EvaluationLevelConfig
from project_module.data_preprocessing import (
    M5Preprocessor,
    RossmannPreprocessor,
    StoreItemPreprocessor,
)
from project_module.data_preprocessing.base import (
    BasePreprocessor,
    PreparedDataset,
)
from project_module.evaluating import (
    compute_wrmsse,
    compute_wrmsse_for_level,
    evaluate_predictions,
)
from project_module.evaluating.wrmsse import get_level_specs
from project_module.models import (
    ChronosForecaster,
    LightGBMForecaster,
    ProphetForecaster,
    Seq2SeqForecaster,
)
from project_module.models.base import ForecastModel, ModelResult
from project_module.timecopilot import (
    DatasetInfo,
    generate_experiment_summary,
    load_evaluation_result,
    run_timecopilot,
    write_experiment_summary,
)
from project_module.timecopilot.azure_openai import ensure_openai_compatible_env
from project_module.timecopilot.model_selection import ModelSelectionLog
from project_module.utils import (
    GPUScheduler,
    allocate_gpus_for_model,
    detect_gpus,
    get_custom_logger,
    run_with_visible_gpus,
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Paths for experiment outputs."""

    root: Path
    run_id: str
    predictions_dir: Path
    metrics_dir: Path
    timecopilot_dir: Path


@dataclass(frozen=True)
class ModelOutput:
    """Standardized model output for evaluation."""

    model_name: str
    predictions: pd.DataFrame
    train_time_sec: float


def resolve_gpu_scheduler(
    config: AppConfig,
    logger: logging.Logger,
) -> tuple[list[int], GPUScheduler | None]:
    """
    Resolve the GPU scheduler based on configuration.

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    tuple[list[int], GPUScheduler | None]
        Available GPU indices and scheduler (if available).
    """
    if not config.gpu.auto_detect:
        logger.info(msg="GPU auto-detect disabled. Using CPU only.")
        return [], None
    available = detect_gpus()
    if not available:
        logger.info(msg="No GPUs detected. Using CPU only.")
        return [], None
    logger.info(msg=f"Detected GPUs: {available}")
    return available, GPUScheduler(available=available)


def run_baseline_model(
    model: ForecastModel,
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
) -> ModelResult:
    """
    Run a baseline model fit/predict cycle.

    Parameters
    ----------
    model : ForecastModel
        Baseline model instance.
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
    return model.fit_predict(
        train_df=train_df,
        horizon=horizon,
        freq=freq,
    )


def build_experiment_paths(
    output_dir: Path,
    experiment_name: str,
    run_id: str | None = None,
) -> ExperimentPaths:
    """
    Create directories for an experiment run.

    Parameters
    ----------
    output_dir : Path
        Base output directory.
    experiment_name : str
        Experiment name.

    Returns
    -------
    ExperimentPaths
        Paths for outputs.
    """
    env_run_id = os.getenv("EXPERIMENT_RUN_ID")
    resolved_run_id = run_id or env_run_id
    if not resolved_run_id:
        resolved_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_dir / experiment_name / resolved_run_id
    if root.exists() and not env_run_id:
        suffix = 1
        while root.exists():
            resolved_run_id = f"{resolved_run_id}_{suffix}"
            root = output_dir / experiment_name / resolved_run_id
            suffix += 1
    predictions_dir = root / "predictions"
    metrics_dir = root / "metrics"
    timecopilot_dir = root / "timecopilot"
    for directory in [predictions_dir, metrics_dir, timecopilot_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        root=root,
        run_id=resolved_run_id,
        predictions_dir=predictions_dir,
        metrics_dir=metrics_dir,
        timecopilot_dir=timecopilot_dir,
    )


def _is_pid_alive(pid: int) -> bool:
    """Return True if the process with the given PID is alive."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_status(
    status_path: Path,
    *,
    status: str,
    run_id: str,
    pid: int,
    started_at: str,
    completed_at: str | None = None,
    error_message: str | None = None,
) -> None:
    """Write experiment status to disk."""
    payload = {
        "status": status,
        "run_id": run_id,
        "pid": pid,
        "started_at": started_at,
        "completed_at": completed_at,
        "error_message": error_message,
    }
    status_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save predictions to a parquet file.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Predictions data.
    output_path : Path
        Destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path=output_path, index=False)


def save_metrics_summary(
    metrics: list[dict[str, str | float | int | None]],
    output_path: Path,
) -> None:
    """
    Save metrics summary to CSV.

    Parameters
    ----------
    metrics : list[dict[str, object]]
        Metrics rows.
    output_path : Path
        Destination path.
    """
    df = pd.DataFrame(data=metrics)
    df.to_csv(path_or_buf=output_path, index=False)


def select_preprocessor(config: AppConfig) -> PreparedDataset:
    """
    Select and execute the appropriate preprocessor.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    PreparedDataset
        Prepared dataset.
    """
    preprocessor: BasePreprocessor
    if config.experiment.name == "store_item":
        preprocessor = StoreItemPreprocessor(
            dataset_dir=Path(config.paths.dataset_dir),
            dataset_config=config.experiment.dataset,
            preprocessing=config.experiment.preprocessing,
            holdout=config.experiment.holdout,
            forecasting=config.experiment.forecasting,
        )
    elif config.experiment.name == "rossmann":
        preprocessor = RossmannPreprocessor(
            dataset_dir=Path(config.paths.dataset_dir),
            dataset_config=config.experiment.dataset,
            preprocessing=config.experiment.preprocessing,
            holdout=config.experiment.holdout,
            forecasting=config.experiment.forecasting,
        )
    elif config.experiment.name == "m5":
        preprocessor = M5Preprocessor(
            dataset_dir=Path(config.paths.dataset_dir),
            dataset_config=config.experiment.dataset,
            preprocessing=config.experiment.preprocessing,
            holdout=config.experiment.holdout,
            forecasting=config.experiment.forecasting,
        )
    else:
        raise ValueError(f"Unsupported experiment: {config.experiment.name}")
    return preprocessor.prepare()


def run_baseline_models(
    dataset: PreparedDataset,
    config: AppConfig,
    logger: logging.Logger,
    available_gpus: list[int],
    gpu_scheduler: GPUScheduler | None,
) -> list[ModelResult]:
    """
    Train baseline models and generate forecasts.

    Parameters
    ----------
    dataset : PreparedDataset
        Prepared data.
    config : AppConfig
        Application configuration.
    logger : logging.Logger
        Logger instance.
    available_gpus : list[int]
        Available GPU indices.
    gpu_scheduler : GPUScheduler | None
        GPU scheduler instance.

    Returns
    -------
    list[ModelResult]
        Model results.
    """
    models = [
        LightGBMForecaster(),
        ProphetForecaster(),
        Seq2SeqForecaster(),
        ChronosForecaster(),
    ]
    results: list[ModelResult] = []
    for model in models:
        model_name = model.name
        logger.info(msg=f"Running baseline model: {model_name}")
        gpu_ids = allocate_gpus_for_model(
            model_name=model_name,
            scheduler=gpu_scheduler,
            available_gpus=available_gpus,
            strategy=config.gpu.strategy,
        )
        if gpu_ids:
            logger.info(msg=f"Assigned GPUs {gpu_ids} to {model_name}.")
        else:
            logger.info(msg=f"Using CPU for {model_name}.")
        try:
            result = cast(
                ModelResult,
                run_with_visible_gpus(
                    gpu_ids=gpu_ids,
                    run_fn=run_baseline_model,
                    model=model,
                    train_df=dataset.train_df,
                    horizon=config.experiment.forecasting.horizon,
                    freq=dataset.freq,
                ),
            )
        finally:
            if gpu_scheduler and gpu_ids:
                gpu_scheduler.release(gpu_ids=gpu_ids)
        logger.info(msg=f"Completed baseline model: {model_name}")
        results.append(result)
    return results


def expand_timecopilot_outputs(
    timecopilot_predictions: pd.DataFrame,
    timecopilot_ensemble: pd.DataFrame,
    run_time_sec: float,
) -> list[ModelOutput]:
    """
    Expand TimeCopilot predictions into per-model outputs.

    Parameters
    ----------
    timecopilot_predictions : pandas.DataFrame
        TimeCopilot predictions with model column.
    timecopilot_ensemble : pandas.DataFrame
        Ensemble predictions.
    run_time_sec : float
        Total TimeCopilot runtime.

    Returns
    -------
    list[ModelOutput]
        Model outputs.
    """
    outputs: list[ModelOutput] = []
    if not timecopilot_predictions.empty:
        for model_name, group in timecopilot_predictions.groupby(by="model"):
            outputs.append(
                ModelOutput(
                    model_name=model_name,
                    predictions=group[["unique_id", "ds", "yhat"]],
                    train_time_sec=run_time_sec,
                )
            )
    if not timecopilot_ensemble.empty:
        outputs.append(
            ModelOutput(
                model_name="TimeCopilotMedianEnsemble",
                predictions=timecopilot_ensemble[["unique_id", "ds", "yhat"]],
                train_time_sec=run_time_sec,
            )
        )
    return outputs


def filter_level_scores(
    level_scores: dict[str, float],
    evaluation_levels: list[EvaluationLevelConfig] | None,
    model_name: str,
) -> dict[str, float]:
    """
    Filter level scores based on evaluation config.

    Parameters
    ----------
    level_scores : dict[str, float]
        All level scores.
    evaluation_levels : list[EvaluationLevelConfig] | None
        Evaluation level config.
    model_name : str
        Model name.

    Returns
    -------
    dict[str, float]
        Filtered level scores.
    """
    if not evaluation_levels:
        return level_scores
    level_specs = get_level_specs()
    level_map = {index + 1: spec.name for index, spec in enumerate(level_specs)}
    selected: dict[str, float] = {}
    for level_config in evaluation_levels:
        if level_config.models != "all" and model_name not in level_config.models:
            continue
        if level_config.exclude_models and model_name in level_config.exclude_models:
            continue
        if level_config.levels == "all":
            for name, value in level_scores.items():
                selected[name] = value
            continue
        for level in level_config.levels:
            level_name = level_map.get(level)
            if not level_name:
                continue
            if level_name in level_scores:
                selected[level_name] = level_scores[level_name]
    return selected


def run_experiment(config: AppConfig) -> None:
    """
    Execute the configured experiment.

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    """
    output_dir = Path(config.paths.output_dir)
    experiment_paths = build_experiment_paths(
        output_dir=output_dir,
        experiment_name=config.experiment.name,
    )
    log_file_name = f"{config.experiment.name}_{experiment_paths.run_id}.log"
    os.environ["EXPERIMENT_RUN_ID"] = experiment_paths.run_id
    os.environ["EXPERIMENT_LOG_DIR"] = str(Path(config.paths.log_dir))
    os.environ["EXPERIMENT_LOG_FILE"] = log_file_name
    logger = get_custom_logger(
        log_dir=str(Path(config.paths.log_dir)),
        log_file_name=log_file_name,
        logger_name="experiment",
    )
    logger.info(msg=f"Logging to {Path(config.paths.log_dir) / log_file_name}")
    status_path = experiment_paths.root / "status.json"
    os.environ["EXPERIMENT_STATUS_PATH"] = str(status_path)
    started_at = datetime.now().isoformat()
    if status_path.exists():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            existing_pid = int(status_payload.get("pid", -1))
            if (
                status_payload.get("status") == "running"
                and existing_pid != os.getpid()
                and _is_pid_alive(pid=existing_pid)
            ):
                logger.error(
                    msg=(
                        "Another experiment process is already running for "
                        f"{experiment_paths.run_id} (pid={existing_pid})."
                    )
                )
                return
        except (ValueError, OSError):
            logger.warning(msg="Failed to parse existing status.json; continuing.")
    _write_status(
        status_path=status_path,
        status="running",
        run_id=experiment_paths.run_id,
        pid=os.getpid(),
        started_at=started_at,
    )
    available_gpus, gpu_scheduler = resolve_gpu_scheduler(
        config=config,
        logger=logger,
    )
    set_seed(seed=config.seed)
    ensure_openai_compatible_env()
    dataset = select_preprocessor(config=config)
    series_count = dataset.train_df["unique_id"].nunique()
    train_rows = len(dataset.train_df)
    logger.info(
        msg=(f"Prepared dataset with {series_count} series and {train_rows} training rows.")
    )
    baseline_results = run_baseline_models(
        dataset=dataset,
        config=config,
        logger=logger,
        available_gpus=available_gpus,
        gpu_scheduler=gpu_scheduler,
    )
    logger.info(msg="Running TimeCopilot models.")
    timecopilot_result = run_timecopilot(
        train_df=dataset.train_df,
        horizon=config.experiment.forecasting.horizon,
        freq=dataset.freq,
        dataset_name=config.experiment.name,
        available_gpus=available_gpus,
        gpu_scheduler=gpu_scheduler,
        gpu_strategy=config.gpu.strategy,
        logger=logger,
    )
    logger.info(msg="Completed TimeCopilot models.")

    outputs: list[ModelOutput] = [
        ModelOutput(
            model_name=result.model_name,
            predictions=result.predictions,
            train_time_sec=result.train_time_sec,
        )
        for result in baseline_results
    ]
    outputs.extend(
        expand_timecopilot_outputs(
            timecopilot_predictions=timecopilot_result.predictions,
            timecopilot_ensemble=timecopilot_result.ensemble,
            run_time_sec=timecopilot_result.train_time_sec,
        )
    )

    metrics_rows: list[dict[str, str | float | int | None]] = []
    level_rows: list[dict[str, object]] = []
    for output in outputs:
        save_predictions(
            predictions=output.predictions,
            output_path=experiment_paths.predictions_dir / f"{output.model_name}.parquet",
        )
        if config.experiment.evaluation.metric == "wrmsse":
            if dataset.m5_bundle is None:
                raise RuntimeError("M5 evaluation requires M5 bundle data.")
            aggregation_level = config.experiment.dataset.m5_aggregation_level
            if aggregation_level is None:
                result = compute_wrmsse(
                    predictions=output.predictions,
                    actuals=dataset.test_df,
                    bundle=dataset.m5_bundle,
                )
            else:
                result = compute_wrmsse_for_level(
                    predictions=output.predictions,
                    actuals=dataset.test_df,
                    bundle=dataset.m5_bundle,
                    level=aggregation_level,
                )
            metrics_rows.append(
                {
                    "model": output.model_name,
                    "metric": result.wrmsse,
                    "metric_name": "wrmsse",
                    "train_time_sec": output.train_time_sec,
                }
            )
            filtered = filter_level_scores(
                level_scores=result.level_scores,
                evaluation_levels=config.experiment.evaluation_levels,
                model_name=output.model_name,
            )
            for level_name, score in filtered.items():
                level_rows.append(
                    {
                        "model": output.model_name,
                        "level": level_name,
                        "metric": score,
                    }
                )
        else:
            metric_value = evaluate_predictions(
                predictions=output.predictions,
                actuals=dataset.test_df,
                metric_name=config.experiment.evaluation.metric,
            )
            metrics_rows.append(
                {
                    "model": output.model_name,
                    "metric": metric_value,
                    "metric_name": config.experiment.evaluation.metric,
                    "train_time_sec": output.train_time_sec,
                }
            )

    metrics_path = experiment_paths.metrics_dir / "summary.csv"
    save_metrics_summary(metrics=metrics_rows, output_path=metrics_path)
    if level_rows:
        pd.DataFrame(data=level_rows).to_csv(
            path_or_buf=experiment_paths.metrics_dir / "level_scores.csv",
            index=False,
        )

    dataset_info = DatasetInfo(
        name=config.experiment.name,
        series_count=int(dataset.train_df["unique_id"].nunique()),
        horizon=config.experiment.forecasting.horizon,
        freq=dataset.freq,
    )
    evaluation = load_evaluation_result(summary_path=metrics_path)
    selection_log: ModelSelectionLog = timecopilot_result.selection_log
    selection_log.save(
        output_path=experiment_paths.timecopilot_dir / "model_selection.json",
        dataset_info=dataset_info,
        evaluation=evaluation,
    )
    summary = generate_experiment_summary(
        selection_log=selection_log,
        dataset_info=dataset_info,
        evaluation=evaluation,
        use_llm=False,
    )
    write_experiment_summary(
        summary=summary,
        output_path=experiment_paths.timecopilot_dir / "experiment_summary.md",
    )
    logger.info(msg=f"Experiment completed: {experiment_paths.root}")
    _write_status(
        status_path=status_path,
        status="completed",
        run_id=experiment_paths.run_id,
        pid=os.getpid(),
        started_at=started_at,
        completed_at=datetime.now().isoformat(),
    )


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
