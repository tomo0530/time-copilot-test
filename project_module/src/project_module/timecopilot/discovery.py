from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from typing import Iterable

from project_module.timecopilot.model_selection import ModelSelectionEntry

OPTIONAL_MODELS = {
    "TimeGPT",
    "TabPFN",
}
EXCLUDED_MODEL_NAMES = {
    "MedianEnsemble",
}
EXPLICITLY_SKIPPED_MODELS: dict[str, str] = {
    "AutoNHITS": "AutoNHITS is skipped in this evaluation run.",
    "AutoTFT": "AutoTFT is skipped in this evaluation run.",
    "FlowState": "FlowState is skipped in this evaluation run.",
    "GluonTSForecaster": "GluonTSForecaster is a base wrapper class.",
    "ParallelForecaster": "ParallelForecaster is a mixin, not a standalone model.",
    "TiRex": "TiRex is skipped due to CUDA extension build failures.",
}

MODEL_REQUIRED_PARAMS: dict[str, dict[str, object]] = {
    "Chronos": {"repo_id": "amazon/chronos-2"},
    "Moirai": {"context_length": 512},
    "TimesFM": {
        "repo_id": "google/timesfm-2.5-200m-pytorch",
        "context_length": 2048,
        "batch_size": 256,
    },
    "Sundial": {"context_length": 256},
    "Toto": {"context_length": 256},
}

MODEL_REQUIRED_ENV: dict[str, list[str]] = {
    "TimeGPT": ["NIXTLA_API_KEY"],
}

SKIP_CLASS_NAMES = {
    "BaseModel",
    "BaseForecaster",
    "Forecaster",
    "TimeSeriesModel",
}


def resolve_base_class() -> type | None:
    """
    Resolve the base class for TimeCopilot models.

    Returns
    -------
    type | None
        Base class if found.
    """
    candidates = [
        ("timecopilot.models.utils.forecaster", "Forecaster"),
        ("timecopilot.models.utils.forecaster", "BaseForecaster"),
        ("timecopilot.models.base", "BaseModel"),
        ("timecopilot.models.base", "Forecaster"),
        ("timecopilot.models.base", "BaseForecaster"),
        ("timecopilot.models.base", "TimeSeriesModel"),
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(name=module_name)
        except ModuleNotFoundError:
            continue
        base_class = getattr(module, class_name, None)
        if inspect.isclass(base_class):
            return base_class
    return None


def iter_model_classes() -> Iterable[type]:
    """
    Iterate over TimeCopilot model classes.

    Returns
    -------
    Iterable[type]
        Model classes.
    """
    models_pkg = importlib.import_module(name="timecopilot.models")
    base_class = resolve_base_class()
    for module_info in pkgutil.walk_packages(
        path=models_pkg.__path__,
        prefix=f"{models_pkg.__name__}.",
    ):
        try:
            module = importlib.import_module(name=module_info.name)
        except Exception:  # noqa: BLE001
            continue
        for _, obj in inspect.getmembers(object=module, predicate=inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if obj.__name__.startswith("_") or obj.__name__ in SKIP_CLASS_NAMES:
                continue
            if base_class and not issubclass(obj, base_class):
                continue
            if inspect.isabstract(obj):
                continue
            yield obj


def is_model_env_ready(model_name: str) -> tuple[bool, str | None]:
    """
    Check whether required environment variables are present.

    Parameters
    ----------
    model_name : str
        Model name.

    Returns
    -------
    tuple[bool, str | None]
        Tuple of readiness and reason.
    """
    required = MODEL_REQUIRED_ENV.get(model_name)
    if not required:
        return True, None
    missing = [key for key in required if not os.getenv(key=key)]
    if missing:
        return False, f"Missing env vars: {', '.join(missing)}"
    return True, None


def instantiate_model(model_class: type) -> object:
    """
    Instantiate a TimeCopilot model class.

    Parameters
    ----------
    model_class : type
        Model class.

    Returns
    -------
    object
        Instantiated model.
    """
    params = MODEL_REQUIRED_PARAMS.get(model_class.__name__, {})
    return model_class(**params)


def discover_timecopilot_models() -> tuple[
    list[object],
    list[ModelSelectionEntry],
    list[ModelSelectionEntry],
]:
    """
    Discover and instantiate available TimeCopilot models.

    Returns
    -------
    tuple[list[object], list[ModelSelectionEntry], list[ModelSelectionEntry]]
        Instantiated models, skipped entries, and failed entries.
    """
    models: list[object] = []
    skipped: list[ModelSelectionEntry] = []
    failed: list[ModelSelectionEntry] = []
    for model_class in iter_model_classes():
        model_name = model_class.__name__
        if model_name in EXPLICITLY_SKIPPED_MODELS:
            skipped.append(
                ModelSelectionEntry(
                    model_name=model_name,
                    status="skipped",
                    reason_code="explicitly_skipped",
                    stage="discovery",
                    error_message=EXPLICITLY_SKIPPED_MODELS[model_name],
                )
            )
            continue
        if model_name in EXCLUDED_MODEL_NAMES:
            skipped.append(
                ModelSelectionEntry(
                    model_name=model_name,
                    status="skipped",
                    reason_code="excluded",
                    stage="discovery",
                    error_message="Excluded from automatic selection.",
                )
            )
            continue
        ready, reason = is_model_env_ready(model_name=model_name)
        if not ready:
            skipped.append(
                ModelSelectionEntry(
                    model_name=model_name,
                    status="skipped",
                    reason_code="environment_not_ready",
                    stage="discovery",
                    error_message=reason,
                )
            )
            continue
        try:
            models.append(instantiate_model(model_class=model_class))
        except Exception as exc:  # noqa: BLE001
            if model_name in OPTIONAL_MODELS:
                skipped.append(
                    ModelSelectionEntry(
                        model_name=model_name,
                        status="skipped",
                        reason_code="optional_model_unavailable",
                        stage="instantiation",
                        error_message=str(exc),
                    )
                )
            else:
                failed.append(
                    ModelSelectionEntry(
                        model_name=model_name,
                        status="failed",
                        reason_code="instantiation_failed",
                        stage="instantiation",
                        error_message=str(exc),
                    )
                )
    return models, skipped, failed
