from __future__ import annotations

import os
from pathlib import Path

import pytest

from project_module.data_download.kaggle_downloader import (
    find_default_token_path,
    find_zip_file,
    has_extracted_files,
    parse_legacy_credentials_from_token,
    prepare_kaggle_auth,
    resolve_token_path,
)


def test_parse_legacy_credentials_from_token_colon_format() -> None:
    credentials = parse_legacy_credentials_from_token(token="user:key")
    assert credentials.username == "user"
    assert credentials.key == "key"


def test_parse_legacy_credentials_from_token_json_format() -> None:
    token = '{"username": "user", "key": "key"}'
    credentials = parse_legacy_credentials_from_token(token=token)
    assert credentials.username == "user"
    assert credentials.key == "key"


def test_parse_legacy_credentials_from_token_env_json_format() -> None:
    token = '{"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"}'
    credentials = parse_legacy_credentials_from_token(token=token)
    assert credentials.username == "user"
    assert credentials.key == "key"


def test_parse_legacy_credentials_from_token_access_token_returns_none() -> None:
    assert parse_legacy_credentials_from_token(token="opaque_token") is None


def test_parse_legacy_credentials_from_token_invalid_json_raises() -> None:
    with pytest.raises(ValueError):
        parse_legacy_credentials_from_token(token="{invalid_json}")


def test_resolve_token_path_expands_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_file = tmp_path / ".kaggle" / "access_token"
    token_file.parent.mkdir()
    token_file.write_text("token", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = resolve_token_path(token="~/.kaggle/access_token")
    assert resolved == str(token_file.resolve())


def test_find_default_token_path_returns_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_file = tmp_path / ".kaggle" / "access_token"
    token_file.parent.mkdir()
    token_file.write_text("token", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = find_default_token_path()
    assert resolved == str(token_file.resolve())


def test_prepare_kaggle_auth_uses_token_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_file = tmp_path / ".kaggle" / "access_token"
    token_file.parent.mkdir()
    token_file.write_text("token", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text('KAGGLE_API_TOKEN="~/.kaggle/access_token"\n', encoding="utf-8")

    monkeypatch.setenv("HOME", str(tmp_path))
    original_token = os.environ.pop("KAGGLE_API_TOKEN", None)
    original_user = os.environ.pop("KAGGLE_USERNAME", None)
    original_key = os.environ.pop("KAGGLE_KEY", None)

    try:
        prepare_kaggle_auth(env_path=env_file)
        assert os.environ["KAGGLE_API_TOKEN"] == str(token_file.resolve())
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)
        if original_token is not None:
            os.environ["KAGGLE_API_TOKEN"] = original_token
        if original_user is not None:
            os.environ["KAGGLE_USERNAME"] = original_user
        if original_key is not None:
            os.environ["KAGGLE_KEY"] = original_key


def test_prepare_kaggle_auth_uses_default_token_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_file = tmp_path / ".kaggle" / "access_token"
    token_file.parent.mkdir()
    token_file.write_text("token", encoding="utf-8")
    env_file = tmp_path / "missing.env"

    monkeypatch.setenv("HOME", str(tmp_path))
    original_token = os.environ.pop("KAGGLE_API_TOKEN", None)
    original_user = os.environ.pop("KAGGLE_USERNAME", None)
    original_key = os.environ.pop("KAGGLE_KEY", None)

    try:
        prepare_kaggle_auth(env_path=env_file)
        assert os.environ["KAGGLE_API_TOKEN"] == str(token_file.resolve())
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)
        if original_token is not None:
            os.environ["KAGGLE_API_TOKEN"] = original_token
        if original_user is not None:
            os.environ["KAGGLE_USERNAME"] = original_user
        if original_key is not None:
            os.environ["KAGGLE_KEY"] = original_key


def test_has_extracted_files_ignores_gitkeep(tmp_path: Path) -> None:
    target_dir = tmp_path / "dataset"
    assert has_extracted_files(target_dir=target_dir) is False

    target_dir.mkdir()
    (target_dir / ".gitkeep").write_text("", encoding="utf-8")
    assert has_extracted_files(target_dir=target_dir) is False

    (target_dir / "train.csv").write_text("data", encoding="utf-8")
    assert has_extracted_files(target_dir=target_dir) is True


def test_find_zip_file_expected_name(tmp_path: Path) -> None:
    competition = "sample-competition"
    zip_path = tmp_path / f"{competition}.zip"
    zip_path.write_text("data", encoding="utf-8")

    found = find_zip_file(target_dir=tmp_path, competition=competition)
    assert found == zip_path


def test_find_zip_file_single_fallback(tmp_path: Path) -> None:
    zip_path = tmp_path / "fallback.zip"
    zip_path.write_text("data", encoding="utf-8")

    found = find_zip_file(target_dir=tmp_path, competition="missing")
    assert found == zip_path


def test_find_zip_file_multiple_raises(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_text("data", encoding="utf-8")
    (tmp_path / "b.zip").write_text("data", encoding="utf-8")

    with pytest.raises(RuntimeError):
        find_zip_file(target_dir=tmp_path, competition="multi")
