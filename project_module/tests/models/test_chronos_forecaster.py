from __future__ import annotations

import pandas as pd
import torch
from pytest import MonkeyPatch

from project_module.models import chronos_forecaster


class DummyChronos2Pipeline:
    sentinel: object | None = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        if cls.sentinel is None:
            cls.sentinel = object()
        return cls.sentinel


def test_build_chronos_pipeline_compat_uses_chronos2(
    monkeypatch: MonkeyPatch,
) -> None:
    sentinel = object()
    DummyChronos2Pipeline.sentinel = sentinel
    monkeypatch.setattr(chronos_forecaster, "Chronos2Pipeline", DummyChronos2Pipeline)

    pipeline = chronos_forecaster._build_chronos_pipeline_compat(
        repo_id="amazon/chronos-2",
        device_map="cpu",
    )

    assert pipeline is sentinel


def test_build_context_tensor_adds_variates_dim() -> None:
    dates = pd.date_range(start="2020-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {
            "unique_id": ["A"] * 3,
            "ds": dates,
            "y": [1.0, 2.0, 3.0],
        }
    )

    tensor, unique_ids = chronos_forecaster.build_context_tensor(
        train_df=frame,
        context_length=2,
    )

    assert tensor.shape == (1, 1, 2)
    assert unique_ids == ["A"]


def test_fit_predict_stacks_quantiles(monkeypatch: MonkeyPatch) -> None:
    class DummyChronos2Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            return cls()

        def predict_quantiles(self, inputs, prediction_length, quantile_levels):  # type: ignore[no-untyped-def]
            quantiles = [torch.tensor([[[1.0], [2.0]]])]
            mean = [torch.tensor([[1.5, 2.5]])]
            return quantiles, mean

    monkeypatch.setattr(chronos_forecaster, "Chronos2Pipeline", DummyChronos2Pipeline)

    config = chronos_forecaster.ChronosConfig(
        repo_id="amazon/chronos-2",
        context_length=2,
        batch_size=1,
        quantile_levels=[0.5],
    )
    forecaster = chronos_forecaster.ChronosForecaster(config=config)
    dates = pd.date_range(start="2020-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {
            "unique_id": ["A"] * 3,
            "ds": dates,
            "y": [1.0, 2.0, 3.0],
        }
    )

    result = forecaster.fit_predict(
        train_df=frame,
        horizon=2,
        freq="D",
    )

    assert result.predictions["yhat"].tolist() == [1.0, 2.0]
