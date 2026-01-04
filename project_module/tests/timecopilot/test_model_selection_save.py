from __future__ import annotations

import json
from pathlib import Path

from project_module.timecopilot.model_selection import (
    DatasetInfo,
    EvaluationResult,
    MetricRow,
    ModelSelectionEntry,
    ModelSelectionLog,
)


def test_model_selection_save_generates_summary(tmp_path: Path) -> None:
    log = ModelSelectionLog(
        dataset_name="store_item",
        selected_models=["ModelA"],
        skipped_models=[
            ModelSelectionEntry(
                model_name="ModelB",
                status="skipped",
                reason_code="missing_dependency",
                stage="discovery",
                error_message="missing",
            ),
        ],
        failed_models=[
            ModelSelectionEntry(
                model_name="ModelC",
                status="failed",
                reason_code="forecast_failed",
                stage="forecast",
                error_message="RuntimeError",
            )
        ],
    )
    dataset_info = DatasetInfo(
        name="store_item",
        series_count=3,
        horizon=7,
        freq="D",
    )
    evaluation = EvaluationResult(
        metric_name="smape",
        rows=[
            MetricRow(model="ModelA", metric=0.123, metric_name="smape"),
        ],
    )
    output_path = tmp_path / "model_selection.json"

    log.save(
        output_path=output_path,
        dataset_info=dataset_info,
        evaluation=evaluation,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]
