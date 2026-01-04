from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf

from project_module.config import AppConfig
from project_module.experiment import run_experiment
from project_module.utils import setup_multiprocessing


def resolve_repo_root() -> Path:
    """
    Resolve the repository root based on this script location.

    Returns
    -------
    Path
        Repository root path.
    """
    return Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def handle_run_experiment(cfg: object) -> None:
    """
    Hydra entry point for running experiments.

    Parameters
    ----------
    cfg : object
        Hydra configuration.
    """
    repo_root = resolve_repo_root()
    load_dotenv(dotenv_path=repo_root / ".env")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("Hydra config must resolve to a dictionary.")
    config = AppConfig.model_validate(obj=config_dict)
    try:
        run_experiment(config=config)
    except Exception as exc:  # noqa: BLE001
        logger = logging.getLogger("custom_logger")
        logger.exception("Experiment failed with an unhandled exception.")
        status_path_value = os.getenv("EXPERIMENT_STATUS_PATH")
        if status_path_value:
            status_path = Path(status_path_value)
            started_at = datetime.now().isoformat()
            run_id = os.getenv("EXPERIMENT_RUN_ID", "unknown")
            if status_path.exists():
                try:
                    existing = json.loads(status_path.read_text(encoding="utf-8"))
                    started_at = existing.get("started_at", started_at)
                    run_id = existing.get("run_id", run_id)
                except ValueError:
                    logger.warning("Failed to parse status.json for failure update.")
            payload = {
                "status": "failed",
                "run_id": run_id,
                "pid": os.getpid(),
                "started_at": started_at,
                "completed_at": datetime.now().isoformat(),
                "error_message": str(exc),
            }
            status_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        raise
        raise


if __name__ == "__main__":
    setup_multiprocessing()
    handle_run_experiment()
