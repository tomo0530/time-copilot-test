from __future__ import annotations

import os

import pytest

from project_module.timecopilot.azure_openai import ensure_openai_compatible_env


def test_ensure_openai_compatible_env_sets_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")

    ensure_openai_compatible_env()

    assert os.environ["OPENAI_BASE_URL"] == "https://example.com/openai/v1/"
    assert os.environ["OPENAI_API_KEY"] == "secret"


def test_ensure_openai_compatible_env_preserves_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://existing.example.com/")
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")

    ensure_openai_compatible_env()

    assert os.environ["OPENAI_BASE_URL"] == "https://existing.example.com/"
    assert os.environ["OPENAI_API_KEY"] == "existing-key"
