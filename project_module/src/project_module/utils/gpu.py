from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Lock, Manager
from typing import Callable, Literal


class GPURequirement(Enum):
    """GPU requirement categories."""

    CPU = "cpu"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


MODEL_GPU_REQUIREMENTS: dict[str, GPURequirement] = {
    "Chronos": GPURequirement.MEDIUM,
    "Moirai": GPURequirement.LARGE,
    "TimesFM": GPURequirement.MEDIUM,
    "Sundial": GPURequirement.SMALL,
    "Toto": GPURequirement.SMALL,
    "TimeGPT": GPURequirement.CPU,
    "TiRex": GPURequirement.CPU,
    "FlowState": GPURequirement.CPU,
    "TabPFN": GPURequirement.CPU,
    "AutoARIMA": GPURequirement.CPU,
    "AutoETS": GPURequirement.CPU,
    "SeasonalNaive": GPURequirement.CPU,
    "Theta": GPURequirement.CPU,
    "Prophet": GPURequirement.CPU,
    "LightGBM": GPURequirement.CPU,
    "Seq2Seq": GPURequirement.MEDIUM,
    "Chronos-2": GPURequirement.MEDIUM,
}


def get_gpu_requirement(model_name: str) -> GPURequirement:
    """
    Get GPU requirement for a model.

    Parameters
    ----------
    model_name : str
        Model name.

    Returns
    -------
    GPURequirement
        Requirement category.
    """
    return MODEL_GPU_REQUIREMENTS.get(model_name, GPURequirement.CPU)


def select_gpu_count(
    available_gpus: list[int],
    requirement: GPURequirement,
    strategy: Literal["hybrid", "single", "multi"],
) -> int:
    """
    Select the number of GPUs for a model based on strategy and requirements.

    Parameters
    ----------
    available_gpus : list[int]
        Available GPU indices.
    requirement : GPURequirement
        Requirement category.
    strategy : Literal["hybrid", "single", "multi"]
        GPU strategy configuration.

    Returns
    -------
    int
        Number of GPUs to allocate.
    """
    if not available_gpus or requirement == GPURequirement.CPU:
        return 0
    if strategy == "single":
        return 1
    if strategy == "multi":
        return len(available_gpus)
    if requirement == GPURequirement.LARGE:
        return len(available_gpus)
    if requirement == GPURequirement.MEDIUM:
        return min(2, len(available_gpus))
    return 1


def allocate_gpus_for_model(
    model_name: str,
    scheduler: "GPUScheduler | None",
    available_gpus: list[int],
    strategy: Literal["hybrid", "single", "multi"],
) -> list[int]:
    """
    Allocate GPUs for a model according to requirements.

    Parameters
    ----------
    model_name : str
        Model name used to resolve GPU requirements.
    scheduler : GPUScheduler | None
        Scheduler instance.
    available_gpus : list[int]
        Available GPU indices.
    strategy : Literal["hybrid", "single", "multi"]
        GPU strategy configuration.

    Returns
    -------
    list[int]
        Allocated GPU indices.
    """
    requirement = get_gpu_requirement(model_name=model_name)
    count = select_gpu_count(
        available_gpus=available_gpus,
        requirement=requirement,
        strategy=strategy,
    )
    if scheduler is None or count == 0:
        return []
    allocated = scheduler.allocate(count=count)
    if len(allocated) < count:
        scheduler.release(gpu_ids=allocated)
        return []
    return allocated


def detect_gpus() -> list[int]:
    """
    Detect available GPU indices using nvidia-smi.

    Returns
    -------
    list[int]
        GPU indices.
    """
    try:
        result = subprocess.run(
            args=["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    indices = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.isdigit():
            indices.append(int(line))
    return indices


@dataclass
class GPUScheduler:
    """Simple GPU scheduler with shared state."""

    available: list[int]

    def __post_init__(self) -> None:
        self._manager = Manager()
        self._available = self._manager.list(self.available)
        self._lock = Lock()

    def allocate(self, count: int) -> list[int]:
        """
        Allocate GPUs.

        Parameters
        ----------
        count : int
            Number of GPUs required.

        Returns
        -------
        list[int]
            Allocated GPU indices.
        """
        with self._lock:
            if count <= 0:
                return []
            if len(self._available) < count:
                return []
            allocated = [self._available.pop(0) for _ in range(count)]
            return allocated

    def release(self, gpu_ids: list[int]) -> None:
        """
        Release GPUs back to the pool.

        Parameters
        ----------
        gpu_ids : list[int]
            GPU indices to release.
        """
        with self._lock:
            for gpu_id in gpu_ids:
                if gpu_id not in self._available:
                    self._available.append(gpu_id)


def run_with_visible_gpus(
    gpu_ids: list[int],
    run_fn: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    """
    Execute a function with CUDA_VISIBLE_DEVICES set.

    Parameters
    ----------
    gpu_ids : list[int]
        GPU indices to expose.
    run_fn : Callable[..., object]
        Function to execute.

    Returns
    -------
    object
        Function result.
    """
    if gpu_ids:
        visible = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    else:
        visible = ""
    previous = os.getenv(key="CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    try:
        return run_fn(*args, **kwargs)
    finally:
        if previous is None:
            try:
                os.environ.pop("CUDA_VISIBLE_DEVICES")
            except KeyError:
                pass
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
