from __future__ import annotations

import numpy as np

from project_module.evaluating.metrics import rmspe, smape


def test_smape_basic() -> None:
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    expected = (
        abs(100.0 - 110.0) / ((abs(100.0) + abs(110.0)) / 2.0)
        + abs(200.0 - 190.0) / ((abs(200.0) + abs(190.0)) / 2.0)
    ) / 2.0
    assert np.isclose(a=smape(y_true=y_true, y_pred=y_pred), b=expected)


def test_rmspe_handles_zero() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    assert np.isnan(rmspe(y_true=y_true, y_pred=y_pred))
