from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetInfo(BaseModel):
    """Dataset information for experiment summaries."""

    name: str
    series_count: int
    horizon: int
    freq: str

    model_config = ConfigDict(extra="forbid")


class ModelSelectionEntry(BaseModel):
    """Model selection entry with status."""

    model_name: str
    status: Literal["success", "skipped", "failed"]
    reason_code: str | None = None
    stage: str | None = None
    error_message: str | None = None

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class MetricRow:
    """Single metric row for an experiment."""

    model: str
    metric: float
    metric_name: str
    train_time_sec: float | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation summary for an experiment."""

    metric_name: str
    rows: list[MetricRow]


class ModelSelectionLog(BaseModel):
    """Log describing TimeCopilot model selection."""

    dataset_name: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    selected_models: list[str]
    skipped_models: list[ModelSelectionEntry]
    failed_models: list[ModelSelectionEntry]
    summary: str | None = None

    model_config = ConfigDict(extra="forbid")

    def generate_summary(
        self,
        dataset_info: DatasetInfo,
        evaluation: EvaluationResult | None = None,
    ) -> str:
        """
        Generate a template-based summary.

        Parameters
        ----------
        dataset_info : DatasetInfo
            Dataset metadata.
        evaluation : EvaluationResult | None
            Optional evaluation summary.

        Returns
        -------
        str
            Generated summary.
        """
        total_models = (
            len(self.selected_models) + len(self.skipped_models) + len(self.failed_models)
        )
        selected = ", ".join(self.selected_models)
        skipped = ", ".join(entry.model_name for entry in self.skipped_models)
        failed = ", ".join(entry.model_name for entry in self.failed_models)
        lines = [
            (
                f"{dataset_info.name}データセット（{dataset_info.series_count}系列、"
                f"{dataset_info.horizon}ステップ予測、頻度 {dataset_info.freq}）に対し、"
                f"TimeCopilotは{len(self.selected_models)}種類のモデル"
                f"（{selected}）を使用しました。"
            )
        ]
        if self.skipped_models:
            lines.append(
                f"{len(self.skipped_models)}モデル（{skipped}）は依存関係または"
                "APIキー未設定のためスキップされました。"
            )
        else:
            lines.append("スキップされたモデルはありません。")
        if self.failed_models:
            lines.append(f"{len(self.failed_models)}モデル（{failed}）は実行中に失敗しました。")
        else:
            lines.append("失敗したモデルはありません。")
        if evaluation:
            best = min(evaluation.rows, key=lambda row: row.metric, default=None)
            if best:
                lines.append(
                    f"評価指標は{evaluation.metric_name}で、"
                    f"最良モデルは{best.model}（{best.metric:.4f}）でした。"
                )
        lines.append(f"対象モデル総数は{total_models}です。")
        return " ".join(lines)

    def save(
        self,
        output_path: Path,
        dataset_info: DatasetInfo,
        evaluation: EvaluationResult | None = None,
    ) -> None:
        """
        Save the selection log to JSON.

        Parameters
        ----------
        output_path : Path
            Output path for JSON.
        dataset_info : DatasetInfo
            Dataset metadata.
        evaluation : EvaluationResult | None
            Optional evaluation summary.
        """
        if not self.summary:
            self.summary = self.generate_summary(
                dataset_info=dataset_info,
                evaluation=evaluation,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            data=self.model_dump_json(indent=2),
            encoding="utf-8",
        )
