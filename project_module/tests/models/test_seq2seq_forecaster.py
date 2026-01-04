from __future__ import annotations

import os

from pytest import MonkeyPatch

from project_module.models import seq2seq_forecaster


def _always_true(port: int) -> bool:
    return True


def _always_false(port: int) -> bool:
    return False


def _fixed_port_12000() -> int:
    return 12000


def _fixed_port_13000() -> int:
    return 13000


def test_resolve_master_port_uses_env_when_available(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setattr(seq2seq_forecaster, "_is_port_available", _always_true)
    monkeypatch.setattr(seq2seq_forecaster, "_get_free_port", _fixed_port_12000)
    port = seq2seq_forecaster._resolve_master_port()
    assert port == 12345
    assert os.environ["MASTER_PORT"] == "12345"


def test_resolve_master_port_falls_back_on_invalid_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MASTER_PORT", "not-a-number")
    monkeypatch.setattr(seq2seq_forecaster, "_get_free_port", _fixed_port_12000)
    port = seq2seq_forecaster._resolve_master_port()
    assert port == 12000
    assert os.environ["MASTER_PORT"] == "12000"


def test_resolve_master_port_replaces_used_port(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setattr(seq2seq_forecaster, "_is_port_available", _always_false)
    monkeypatch.setattr(seq2seq_forecaster, "_get_free_port", _fixed_port_13000)
    port = seq2seq_forecaster._resolve_master_port()
    assert port == 13000
    assert os.environ["MASTER_PORT"] == "13000"
