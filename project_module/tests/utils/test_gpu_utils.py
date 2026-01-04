from __future__ import annotations

from project_module.utils.gpu import (
    GPURequirement,
    GPUScheduler,
    allocate_gpus_for_model,
    select_gpu_count,
)


def test_select_gpu_count_hybrid() -> None:
    available = [0, 1, 2]
    assert (
        select_gpu_count(
            available_gpus=available,
            requirement=GPURequirement.CPU,
            strategy="hybrid",
        )
        == 0
    )
    assert (
        select_gpu_count(
            available_gpus=available,
            requirement=GPURequirement.SMALL,
            strategy="hybrid",
        )
        == 1
    )
    assert (
        select_gpu_count(
            available_gpus=available,
            requirement=GPURequirement.MEDIUM,
            strategy="hybrid",
        )
        == 2
    )
    assert (
        select_gpu_count(
            available_gpus=available,
            requirement=GPURequirement.LARGE,
            strategy="hybrid",
        )
        == 3
    )


def test_select_gpu_count_single_multi() -> None:
    available = [0, 1]
    assert (
        select_gpu_count(
            available_gpus=available,
            requirement=GPURequirement.SMALL,
            strategy="single",
        )
        == 1
    )
    assert (
        select_gpu_count(
            available_gpus=available,
            requirement=GPURequirement.SMALL,
            strategy="multi",
        )
        == 2
    )


def test_allocate_gpus_for_model() -> None:
    available = [0, 1]
    scheduler = GPUScheduler(available=available)
    allocated = allocate_gpus_for_model(
        model_name="Chronos",
        scheduler=scheduler,
        available_gpus=available,
        strategy="hybrid",
    )
    assert len(allocated) == 2
    scheduler.release(gpu_ids=allocated)

    cpu_allocated = allocate_gpus_for_model(
        model_name="Prophet",
        scheduler=scheduler,
        available_gpus=available,
        strategy="hybrid",
    )
    assert cpu_allocated == []
