from __future__ import annotations

from pathlib import Path

import pandas as pd

from project_module.timecopilot.summary import load_evaluation_result


def test_load_evaluation_result_reads_metric_name(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    df = pd.DataFrame(
        data=[
            {
                "model": "ModelA",
                "metric": 0.5,
                "metric_name": "smape",
                "train_time_sec": 12.3,
            },
            {
                "model": "ModelB",
                "metric": 0.6,
                "metric_name": "smape",
                "train_time_sec": 9.8,
            },
        ]
    )
    df.to_csv(path_or_buf=summary_path, index=False)

    result = load_evaluation_result(summary_path=summary_path)

    assert result.metric_name == "smape"
    assert len(result.rows) == 2
    assert result.rows[0].model == "ModelA"
