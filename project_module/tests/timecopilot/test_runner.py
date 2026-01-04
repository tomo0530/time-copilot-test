import pandas as pd

from project_module.timecopilot.runner import normalize_forecast_output


def test_normalize_forecast_output_accepts_timesfm_column() -> None:
    forecast_df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": ["2020-01-01", "2020-01-02"],
            "TimesFM": [1.0, 2.0],
        }
    )

    output = normalize_forecast_output(
        forecast_df=forecast_df,
        model_name="_TimesFMV2_p5",
    )

    assert output["yhat"].tolist() == [1.0, 2.0]
    assert output["model"].unique().tolist() == ["_TimesFMV2_p5"]
