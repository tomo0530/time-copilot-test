from __future__ import annotations

import argparse
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kaggle.api.kaggle_api_extended import KaggleApi

DEFAULT_COMPETITIONS = (
    "demand-forecasting-kernels-only",
    "rossmann-store-sales",
    "m5-forecasting-accuracy",
)


class KaggleCredentials(BaseModel):
    """Kaggle API credentials."""

    username: str = Field(min_length=1)
    key: str = Field(min_length=1)


class KaggleCompetition(BaseModel):
    """Configuration for a Kaggle competition dataset."""

    slug: str = Field(min_length=1)


class KaggleAuthValues(BaseModel):
    """Kaggle authentication values loaded from the environment."""

    api_token: str | None = None
    username: str | None = None
    key: str | None = None


def configure_logging(log_file: Path) -> None:
    """
    Configure application logging to stdout and a log file.

    Parameters
    ----------
    log_file : Path
        Log file path.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def resolve_repo_root(script_path: Path) -> Path:
    """
    Resolve the repository root based on the script path.

    Parameters
    ----------
    script_path : Path
        Path to the current script.

    Returns
    -------
    Path
        Repository root directory.
    """
    for parent in script_path.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    return script_path.parents[2]


def load_kaggle_env_values(env_path: Path) -> KaggleAuthValues:
    """
    Load Kaggle authentication values from a .env file.

    Parameters
    ----------
    env_path : Path
        Path to the .env file.

    Returns
    -------
    KaggleAuthValues
        Authentication values loaded from the environment. If the .env file
        does not exist, values are read from the current environment only.
    """

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    return KaggleAuthValues(
        api_token=os.getenv(key="KAGGLE_API_TOKEN"),
        username=os.getenv(key="KAGGLE_USERNAME"),
        key=os.getenv(key="KAGGLE_KEY"),
    )


def parse_legacy_credentials_from_token(
    token: str,
) -> KaggleCredentials | None:
    """
    Parse legacy Kaggle credentials from a token string.

    Parameters
    ----------
    token : str
        Token string. Supported formats:
        - "username:key"
        - JSON string with "username" and "key" fields.

    Returns
    -------
    KaggleCredentials | None
        Parsed credentials if a legacy format is detected.

    Raises
    ------
    ValueError
        If the token starts with JSON but is invalid or missing fields.
    """

    cleaned = normalize_env_value(value=token)
    if Path(cleaned).exists():
        return None
    if cleaned.startswith("{"):
        payload = parse_token_json(token=cleaned)
        if "username" in payload and "key" in payload:
            return KaggleCredentials(
                username=str(payload["username"]),
                key=str(payload["key"]),
            )
        if "KAGGLE_USERNAME" in payload and "KAGGLE_KEY" in payload:
            return KaggleCredentials(
                username=str(payload["KAGGLE_USERNAME"]),
                key=str(payload["KAGGLE_KEY"]),
            )
        raise ValueError("Token JSON must include username/key fields.")

    if ":" in cleaned:
        username, key = cleaned.split(":", 1)
        if username.strip() and key.strip():
            return KaggleCredentials(username=username.strip(), key=key.strip())
        return None

    return None


def strip_wrapping_quotes(value: str) -> str:
    """
    Strip a single pair of wrapping quotes from a string.

    Parameters
    ----------
    value : str
        Input string.

    Returns
    -------
    str
        Unwrapped string if wrapped in single or double quotes.
    """

    if len(value) < 2:
        return value
    if value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def normalize_env_value(value: str) -> str:
    """
    Normalize an environment value to a clean token string.

    Parameters
    ----------
    value : str
        Raw value from the environment.

    Returns
    -------
    str
        Normalized value with smart quotes removed and wrapping quotes stripped.
    """

    cleaned = value.strip()
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = cleaned.replace("\uff1a", ":")
    return strip_wrapping_quotes(value=cleaned)


def resolve_token_path(token: str) -> str | None:
    """
    Resolve a token file path if the token points to a local file.

    Parameters
    ----------
    token : str
        Candidate token or path string.

    Returns
    -------
    str | None
        Expanded absolute path if the token points to an existing file.
    """

    candidate = Path(token).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return None


def find_default_token_path() -> str | None:
    """
    Find a default Kaggle access token file path.

    Returns
    -------
    str | None
        Expanded absolute path if a token file exists.
    """

    candidates = [
        Path("~/.kaggle/access_token").expanduser(),
        Path("~/.kaggle/access_token.txt").expanduser(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def parse_token_json(token: str) -> dict[str, str]:
    """
    Parse token JSON into a dictionary.

    Parameters
    ----------
    token : str
        JSON string.

    Returns
    -------
    dict[str, str]
        Parsed key-value mapping.

    Raises
    ------
    ValueError
        If the JSON cannot be parsed into an object.
    """

    try:
        payload = json.loads(s=token)
    except json.JSONDecodeError as exc:
        raise ValueError("KAGGLE_API_TOKEN JSON is invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("KAGGLE_API_TOKEN JSON must be an object.")
    return {str(key): str(value) for key, value in payload.items()}


def set_kaggle_env(credentials: KaggleCredentials) -> None:
    """
    Set Kaggle API credentials into the process environment.

    Parameters
    ----------
    credentials : KaggleCredentials
        Kaggle API credentials.
    """

    os.environ["KAGGLE_USERNAME"] = credentials.username
    os.environ["KAGGLE_KEY"] = credentials.key


def prepare_kaggle_auth(env_path: Path) -> None:
    """
    Prepare Kaggle authentication environment variables.

    Parameters
    ----------
    env_path : Path
        Path to the .env file containing Kaggle credentials.

    Raises
    ------
    RuntimeError
        If no valid Kaggle credentials are available.
    """

    auth_values = load_kaggle_env_values(env_path=env_path)
    if auth_values.api_token:
        token = normalize_env_value(value=auth_values.api_token)
        token_path = resolve_token_path(token=token)
        if token_path:
            os.environ["KAGGLE_API_TOKEN"] = token_path
            return
        legacy_credentials = parse_legacy_credentials_from_token(token=token)
        if legacy_credentials is not None:
            set_kaggle_env(credentials=legacy_credentials)
            os.environ.pop("KAGGLE_API_TOKEN", None)
            return
        os.environ["KAGGLE_API_TOKEN"] = token
        return

    default_token_path = find_default_token_path()
    if default_token_path:
        os.environ["KAGGLE_API_TOKEN"] = default_token_path
        return

    if auth_values.username and auth_values.key:
        credentials = KaggleCredentials(
            username=normalize_env_value(value=auth_values.username),
            key=normalize_env_value(value=auth_values.key),
        )
        set_kaggle_env(credentials=credentials)
        return

    raise RuntimeError(
        "Kaggle credentials are missing. Set KAGGLE_API_TOKEN, create "
        "~/.kaggle/access_token, or set KAGGLE_USERNAME and KAGGLE_KEY."
    )


def ensure_directory(path: Path) -> None:
    """
    Ensure that a directory exists.

    Parameters
    ----------
    path : Path
        Directory path to create.
    """

    path.mkdir(parents=True, exist_ok=True)


def has_extracted_files(target_dir: Path) -> bool:
    """
    Check if the target directory already contains extracted data files.

    Parameters
    ----------
    target_dir : Path
        Directory to inspect.

    Returns
    -------
    bool
        True if non-placeholder files exist.
    """

    if not target_dir.exists():
        return False
    for entry in target_dir.iterdir():
        if entry.name == ".gitkeep":
            continue
        return True
    return False


def find_zip_file(target_dir: Path, competition: str) -> Path:
    """
    Locate the downloaded zip file for a competition.

    Parameters
    ----------
    target_dir : Path
        Directory containing the downloaded zip file.
    competition : str
        Competition slug.

    Returns
    -------
    Path
        Path to the zip file.

    Raises
    ------
    FileNotFoundError
        If no zip file is found.
    RuntimeError
        If multiple zip files are found.
    """

    expected = target_dir / f"{competition}.zip"
    if expected.exists():
        return expected

    zip_files = sorted(target_dir.glob(pattern="*.zip"))
    if len(zip_files) == 1:
        return zip_files[0]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found after download for {competition}.")
    raise RuntimeError(f"Multiple zip files found for {competition}: {zip_files}")


def extract_zip(zip_path: Path, extract_dir: Path, remove_zip: bool) -> None:
    """
    Extract a zip archive and optionally remove it.

    Parameters
    ----------
    zip_path : Path
        Path to the zip archive.
    extract_dir : Path
        Destination directory for extraction.
    remove_zip : bool
        Whether to delete the zip file after extraction.
    """

    with zipfile.ZipFile(file=zip_path, mode="r") as archive:
        archive.extractall(path=extract_dir)
    if remove_zip:
        zip_path.unlink()


def download_competition_dataset(
    api: KaggleApi,
    competition: KaggleCompetition,
    dataset_dir: Path,
    force: bool,
    remove_zip: bool,
) -> None:
    """
    Download and extract a competition dataset.

    Parameters
    ----------
    api : KaggleApi
        Authenticated Kaggle API client.
    competition : KaggleCompetition
        Competition configuration.
    dataset_dir : Path
        Base dataset directory.
    force : bool
        Whether to force re-download if data exists.
    remove_zip : bool
        Whether to remove the zip after extraction.
    """

    logger = logging.getLogger(name=__name__)
    target_dir = dataset_dir / competition.slug
    ensure_directory(path=target_dir)

    if not force and has_extracted_files(target_dir=target_dir):
        logger.info("Skip %s (already extracted).", competition.slug)
        return

    logger.info("Downloading %s...", competition.slug)
    api.competition_download_files(
        competition=competition.slug,
        path=str(target_dir),
        quiet=False,
        force=force,
    )

    zip_path = find_zip_file(target_dir=target_dir, competition=competition.slug)
    logger.info("Extracting %s...", zip_path.name)
    extract_zip(zip_path=zip_path, extract_dir=target_dir, remove_zip=remove_zip)


def download_all_competitions(
    dataset_dir: Path,
    env_path: Path,
    force: bool,
    remove_zip: bool,
) -> None:
    """
    Download all configured Kaggle competition datasets.

    Parameters
    ----------
    dataset_dir : Path
        Base dataset directory.
    env_path : Path
        Path to the .env file containing Kaggle credentials.
    force : bool
        Whether to force re-download even if files exist.
    remove_zip : bool
        Whether to remove zip archives after extraction.
    """

    prepare_kaggle_auth(env_path=env_path)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    ensure_directory(path=dataset_dir)
    for slug in DEFAULT_COMPETITIONS:
        competition = KaggleCompetition(slug=slug)
        download_competition_dataset(
            api=api,
            competition=competition,
            dataset_dir=dataset_dir,
            force=force,
            remove_zip=remove_zip,
        )


def handle_cli() -> int:
    """
    CLI entry point for downloading Kaggle datasets.

    Returns
    -------
    int
        Process exit code.
    """

    # fmt: off
    parser = argparse.ArgumentParser(
        description=(
            "Download Kaggle competition datasets into the local dataset "
            "directory."
        )
    )
    # fmt: on
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: repo_root/logs/kaggle_download.log).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Destination directory (default: repo_root/dataset).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: repo_root/.env).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if extracted files exist.",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root(script_path=Path(__file__).resolve())
    dataset_dir = args.dataset_dir or (repo_root / "dataset")
    env_path = args.env_file or (repo_root / ".env")
    log_file = args.log_file or (repo_root / "logs" / "kaggle_download.log")

    configure_logging(log_file=log_file)

    try:
        download_all_competitions(
            dataset_dir=dataset_dir,
            env_path=env_path,
            force=args.force,
            remove_zip=True,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        logging.getLogger(name=__name__).error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(handle_cli())
