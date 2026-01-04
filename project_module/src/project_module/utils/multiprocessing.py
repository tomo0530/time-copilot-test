from __future__ import annotations

import multiprocessing as mp


def setup_multiprocessing() -> None:
    """
    Configure multiprocessing to use spawn.
    """
    try:
        mp.set_start_method(method="spawn", force=True)
    except RuntimeError:
        return
