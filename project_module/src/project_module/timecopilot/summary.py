from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from project_module.timecopilot.azure_openai import create_summary_llm_client
from project_module.timecopilot.model_selection import (
    DatasetInfo,
    EvaluationResult,
    MetricRow,
    ModelSelectionLog,
)


def load_evaluation_result(summary_path: Path) -> EvaluationResult:
    """
    Load evaluation results from summary.csv.

    Parameters
    ----------
    summary_path : Path
        Path to summary.csv.

    Returns
    -------
    EvaluationResult
        Parsed evaluation results.
    """
    df = pd.read_csv(filepath_or_buffer=summary_path)
    rows = [
        MetricRow(
            model=str(row["model"]),
            metric=float(row["metric"]),
            metric_name=str(row["metric_name"]),
            train_time_sec=(float(row["train_time_sec"]) if "train_time_sec" in row else None),
        )
        for _, row in df.iterrows()
    ]
    metric_name = rows[0].metric_name if rows else "unknown"
    return EvaluationResult(metric_name=metric_name, rows=rows)


def generate_experiment_summary(
    selection_log: ModelSelectionLog,
    dataset_info: DatasetInfo,
    evaluation: EvaluationResult | None = None,
    use_llm: bool = False,
) -> str:
    """
    Generate an experiment summary report.

    Parameters
    ----------
    selection_log : ModelSelectionLog
        Model selection log.
    dataset_info : DatasetInfo
        Dataset metadata.
    evaluation : EvaluationResult | None
        Evaluation results.
    use_llm : bool
        Whether to generate an LLM-based summary.

    Returns
    -------
    str
        Summary text.
    """
    if use_llm:
        return generate_llm_summary(
            selection_log=selection_log,
            dataset_info=dataset_info,
            evaluation=evaluation,
        )
    return selection_log.generate_summary(
        dataset_info=dataset_info,
        evaluation=evaluation,
    )


def generate_llm_summary(
    selection_log: ModelSelectionLog,
    dataset_info: DatasetInfo,
    evaluation: EvaluationResult | None = None,
) -> str:
    """
    Generate a summary using Azure OpenAI.

    Parameters
    ----------
    selection_log : ModelSelectionLog
        Model selection log.
    dataset_info : DatasetInfo
        Dataset metadata.
    evaluation : EvaluationResult | None
        Evaluation results.

    Returns
    -------
    str
        LLM-generated summary.
    """
    deployment = os.getenv(key="AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT_NAME is required for LLM summaries.")
    client = create_summary_llm_client()
    prompt = selection_log.generate_summary(
        dataset_info=dataset_info,
        evaluation=evaluation,
    )
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": (f"次の要約をより自然で読みやすい日本語にしてください。\n\n{prompt}"),
            }
        ],
    )
    content = completion.choices[0].message.content or ""
    return content.strip()


def write_experiment_summary(summary: str, output_path: Path) -> None:
    """
    Write experiment summary to disk.

    Parameters
    ----------
    summary : str
        Summary text.
    output_path : Path
        Destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data=summary, encoding="utf-8")
