from __future__ import annotations

from project_module.timecopilot.model_selection import (
    DatasetInfo,
    EvaluationResult,
    MetricRow,
    ModelSelectionEntry,
    ModelSelectionLog,
)


def test_model_selection_summary_includes_dataset() -> None:
    log = ModelSelectionLog(
        dataset_name="store_item",
        selected_models=["ModelA", "ModelB"],
        skipped_models=[
            ModelSelectionEntry(
                model_name="ModelC",
                status="skipped",
                reason_code="missing_dependency",
                stage="discovery",
                error_message="missing",
            ),
        ],
        failed_models=[
            ModelSelectionEntry(
                model_name="ModelD",
                status="failed",
                reason_code="forecast_failed",
                stage="forecast",
                error_message="RuntimeError",
            )
        ],
    )
    dataset_info = DatasetInfo(
        name="store_item",
        series_count=5,
        horizon=10,
        freq="D",
    )
    evaluation = EvaluationResult(
        metric_name="smape",
        rows=[
            MetricRow(model="ModelA", metric=0.1, metric_name="smape"),
        ],
    )
    summary = log.generate_summary(dataset_info=dataset_info, evaluation=evaluation)
    assert "store_item" in summary
    assert "5系列" in summary
    assert "ModelA" in summary
    assert "ModelD" in summary
