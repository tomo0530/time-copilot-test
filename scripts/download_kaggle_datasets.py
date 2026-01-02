from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """
    Run the Kaggle dataset download CLI.

    Returns
    -------
    int
        Process exit code.
    """

    repo_root = Path(__file__).resolve().parents[1]
    command = [
        "uv",
        "run",
        "--project",
        "project_module",
        "python",
        "-m",
        "project_module.data_download.kaggle_downloader",
        *sys.argv[1:],
    ]
    result = subprocess.run(args=command, check=False, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
