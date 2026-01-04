from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from project_module.models.base import ForecastModel, ModelResult
from project_module.utils import get_custom_logger


def _get_experiment_logger() -> logging.Logger:
    """
    Return a logger configured for experiment logs.

    Returns
    -------
    logging.Logger
        Logger instance for experiment logs.
    """
    logger = logging.getLogger("custom_logger")
    if logger.handlers:
        return logger
    log_dir = os.getenv("EXPERIMENT_LOG_DIR")
    log_file = os.getenv("EXPERIMENT_LOG_FILE")
    if log_dir and log_file:
        return get_custom_logger(
            log_dir=log_dir,
            log_file_name=log_file,
            logger_name="custom_logger",
        )
    return logger


@dataclass(frozen=True)
class Seq2SeqConfig:
    """Configuration for the Seq2Seq forecaster."""

    context_length: int
    hidden_size: int
    epochs: int
    batch_size: int
    learning_rate: float


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of fixed-length sequences for seq2seq training."""

    def __init__(
        self,
        series_values: list[npt.NDArray[np.float64]],
        context_length: int,
        horizon: int,
    ) -> None:
        self._inputs: list[npt.NDArray[np.float32]] = []
        self._targets: list[npt.NDArray[np.float32]] = []
        for values in series_values:
            if len(values) < context_length + horizon:
                continue
            input_seq = values[-(context_length + horizon) : -horizon]
            target_seq = values[-horizon:]
            self._inputs.append(input_seq.astype(dtype=np.float32))
            self._targets.append(target_seq.astype(dtype=np.float32))
        if not self._inputs:
            raise ValueError("Seq2Seq dataset has no valid sequences.")

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.tensor(data=self._inputs[index])
        inputs = inputs.unsqueeze(dim=-1)
        targets = torch.tensor(data=self._targets[index])
        return inputs, targets


class SimpleSeq2Seq(nn.Module):
    """Minimal LSTM encoder with linear decoder."""

    def __init__(self, hidden_size: int, horizon: int) -> None:
        super().__init__()
        self._lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self._decoder = nn.Linear(hidden_size, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self._lstm(inputs)
        hidden_last: torch.Tensor = hidden[-1]
        return cast(torch.Tensor, self._decoder(hidden_last))


def build_series_values(
    train_df: pd.DataFrame,
) -> list[npt.NDArray[np.float64]]:
    """
    Build per-series value arrays sorted by timestamp.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data.

    Returns
    -------
    list[numpy.ndarray]
        Series value arrays.
    """
    values: list[npt.NDArray[np.float64]] = []
    for _, series_df in train_df.groupby(by="unique_id"):
        series_df = series_df.copy()
        series_df["ds"] = pd.to_datetime(arg=series_df["ds"])
        series_values = series_df.sort_values(by="ds")["y"].to_numpy(dtype=float)
        values.append(series_values)
    return values


def train_seq2seq(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    log_rank: int | None = None,
) -> None:
    """
    Train the seq2seq model.

    Parameters
    ----------
    model : SimpleSeq2Seq
        Model to train.
    data_loader : DataLoader
        Data loader.
    device : torch.device
        Target device.
    epochs : int
        Number of epochs.
    learning_rate : float
        Learning rate.
    log_rank : int | None
        Rank for logging. Logs only when None or rank 0.
    """
    logger = _get_experiment_logger()
    should_log = log_rank is None or log_rank == 0
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )
    loss_fn = nn.MSELoss()
    model.to(device=device)
    model.train()
    train_start = time.perf_counter()
    if should_log:
        logger.info(
            msg=(
                "Seq2Seq training start: "
                f"epochs={epochs}, batches={len(data_loader)}, "
                f"batch_size={data_loader.batch_size}."
            )
        )
    for epoch_index in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1
        if should_log:
            avg_loss = epoch_loss / batch_count if batch_count else float("nan")
            logger.info(msg=(f"Seq2Seq epoch {epoch_index + 1}/{epochs}: avg_loss={avg_loss:.6f}."))
    if should_log:
        logger.info(msg=f"Seq2Seq training complete in {time.perf_counter() - train_start:.2f}s.")


def _is_port_available(port: int) -> bool:
    """
    Check whether a TCP port can be bound on localhost.

    Parameters
    ----------
    port : int
        Port number to check.

    Returns
    -------
    bool
        True if the port is available.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _get_free_port() -> int:
    """
    Allocate a free TCP port on localhost.

    Returns
    -------
    int
        Free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_master_addr() -> str:
    """
    Resolve the master address for DDP.

    Returns
    -------
    str
        Master address.
    """
    master_addr = os.getenv("MASTER_ADDR")
    if master_addr:
        return master_addr
    master_addr = "127.0.0.1"
    os.environ["MASTER_ADDR"] = master_addr
    return master_addr


def _resolve_master_port() -> int:
    """
    Resolve a usable master port for DDP.

    Returns
    -------
    int
        Master port.
    """
    port_env = os.getenv("MASTER_PORT")
    logger = _get_experiment_logger()
    port: int | None = None
    if port_env:
        try:
            candidate = int(port_env)
        except ValueError:
            logger.warning(
                "Invalid MASTER_PORT '%s'; selecting a free port.",
                port_env,
            )
        else:
            if 0 < candidate <= 65535 and _is_port_available(port=candidate):
                port = candidate
            else:
                logger.warning(
                    "MASTER_PORT %s is unavailable; selecting a free port.",
                    candidate,
                )
    if port is None:
        port = _get_free_port()
    os.environ["MASTER_PORT"] = str(port)
    return port


def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: int) -> None:
    """
    Initialize distributed training.

    Parameters
    ----------
    rank : int
        Process rank.
    world_size : int
        Total number of processes.
    master_addr : str
        Master address for the process group.
    master_port : int
        Master port for the process group.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def cleanup_ddp() -> None:
    """Clean up distributed training state."""
    torch.distributed.destroy_process_group()


def ddp_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    model_state_path: str,
    series_values: list[npt.NDArray[np.float64]],
    config: Seq2SeqConfig,
    horizon: int,
) -> None:
    """
    Train a seq2seq model in a DDP worker.

    Parameters
    ----------
    rank : int
        Process rank.
    world_size : int
        Number of processes.
    master_addr : str
        Master address for the process group.
    master_port : int
        Master port for the process group.
    model_state_path : str
        Path to save the model state.
    series_values : list[numpy.ndarray]
        Series values.
    config : Seq2SeqConfig
        Configuration.
    horizon : int
        Forecast horizon.
    """
    logger = _get_experiment_logger()
    if rank == 0:
        logger.info(
            msg=(
                "Seq2Seq DDP worker start: "
                f"world_size={world_size}, master_addr={master_addr}, "
                f"master_port={master_port}."
            )
        )
    setup_ddp(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    )
    device = torch.device(type="cuda", index=rank)
    dataset = SequenceDataset(
        series_values=series_values,
        context_length=config.context_length,
        horizon=horizon,
    )
    sampler: DistributedSampler[SequenceDataset] = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler=sampler,
    )
    model = SimpleSeq2Seq(hidden_size=config.hidden_size, horizon=horizon)
    model = nn.parallel.DistributedDataParallel(
        model.to(device),
        device_ids=[rank],
    )
    train_seq2seq(
        model=model,
        data_loader=loader,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        log_rank=rank,
    )
    if rank == 0:
        torch.save(obj=model.module.state_dict(), f=model_state_path)
    cleanup_ddp()
    if rank == 0:
        logger.info(msg="Seq2Seq DDP worker completed training.")


def train_seq2seq_distributed(
    series_values: list[npt.NDArray[np.float64]],
    config: Seq2SeqConfig,
    horizon: int,
    model_state_path: Path,
) -> SimpleSeq2Seq:
    """
    Train the seq2seq model with DDP.

    Parameters
    ----------
    series_values : list[numpy.ndarray]
        Series values.
    config : Seq2SeqConfig
        Configuration.
    horizon : int
        Forecast horizon.
    model_state_path : Path
        Path to save the model state.

    Returns
    -------
    SimpleSeq2Seq
        Trained model.
    """
    logger = _get_experiment_logger()
    world_size = torch.cuda.device_count()
    master_addr = _resolve_master_addr()
    master_port = _resolve_master_port()
    logger.info(
        msg=(
            "Seq2Seq DDP launch: "
            f"world_size={world_size}, master_addr={master_addr}, "
            f"master_port={master_port}, series={len(series_values)}, "
            f"epochs={config.epochs}."
        )
    )
    torch.multiprocessing.spawn(  # type: ignore[attr-defined]
        fn=ddp_worker,
        args=(
            world_size,
            master_addr,
            master_port,
            str(model_state_path),
            series_values,
            config,
            horizon,
        ),
        nprocs=world_size,
        join=True,
    )
    logger.info(msg="Seq2Seq DDP training finished; loading state dict.")
    model = SimpleSeq2Seq(hidden_size=config.hidden_size, horizon=horizon)
    model.load_state_dict(state_dict=torch.load(f=model_state_path, map_location="cpu"))
    return model


def train_seq2seq_single(
    series_values: list[npt.NDArray[np.float64]],
    config: Seq2SeqConfig,
    horizon: int,
    device: torch.device,
) -> SimpleSeq2Seq:
    """
    Train seq2seq model on a single process.

    Parameters
    ----------
    series_values : list[numpy.ndarray]
        Series values.
    config : Seq2SeqConfig
        Configuration.
    horizon : int
        Forecast horizon.
    device : torch.device
        Target device.

    Returns
    -------
    SimpleSeq2Seq
        Trained model.
    """
    logger = _get_experiment_logger()
    logger.info(
        msg=(
            "Seq2Seq single-GPU training start: "
            f"series={len(series_values)}, epochs={config.epochs}, "
            f"batch_size={config.batch_size}."
        )
    )
    dataset = SequenceDataset(
        series_values=series_values,
        context_length=config.context_length,
        horizon=horizon,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    model = SimpleSeq2Seq(hidden_size=config.hidden_size, horizon=horizon)
    train_seq2seq(
        model=model,
        data_loader=loader,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        log_rank=0,
    )
    logger.info(msg="Seq2Seq single-GPU training finished.")
    return model


def forecast_seq2seq(
    model: SimpleSeq2Seq,
    train_df: pd.DataFrame,
    context_length: int,
    horizon: int,
    freq: str,
    device: torch.device,
) -> pd.DataFrame:
    """
    Generate forecasts from the trained seq2seq model.

    Parameters
    ----------
    model : SimpleSeq2Seq
        Trained model.
    train_df : pandas.DataFrame
        Training data.
    context_length : int
        Context window length.
    horizon : int
        Forecast horizon.
    freq : str
        Pandas frequency string.
    device : torch.device
        Target device.

    Returns
    -------
    pandas.DataFrame
        Forecasts.
    """
    logger = _get_experiment_logger()
    total_series = train_df["unique_id"].nunique()
    log_interval = max(1, total_series // 10)
    processed = 0
    skipped = 0
    logger.info(
        msg=(
            "Seq2Seq forecasting start: "
            f"series={total_series}, context_length={context_length}, "
            f"horizon={horizon}."
        )
    )
    model = model.to(device=device)
    model.eval()
    predictions: list[pd.DataFrame] = []
    for index, (unique_id, series_df) in enumerate(
        train_df.groupby(by="unique_id"),
        start=1,
    ):
        series_df = series_df.copy()
        series_df["ds"] = pd.to_datetime(arg=series_df["ds"])
        series_df = series_df.sort_values(by="ds")
        values = series_df["y"].to_numpy()
        if len(values) < context_length:
            skipped += 1
            if index % log_interval == 0 or index == total_series:
                logger.info(
                    msg=(
                        "Seq2Seq forecasting progress: "
                        f"{index}/{total_series} series (skipped={skipped})."
                    )
                )
            continue
        context = values[-context_length:].astype(dtype=np.float32)
        inputs = torch.tensor(data=context)
        inputs = inputs.unsqueeze(dim=0)
        inputs = inputs.unsqueeze(dim=-1)
        inputs = inputs.to(device=device)
        with torch.no_grad():
            output = model(inputs).cpu().numpy().reshape(-1)
        future_dates = pd.date_range(
            start=series_df["ds"].max() + pd.tseries.frequencies.to_offset(freq=freq),
            periods=horizon,
            freq=freq,
        )
        series_pred = pd.DataFrame(
            data={
                "unique_id": str(unique_id),
                "ds": future_dates,
                "yhat": output[:horizon],
            }
        )
        predictions.append(series_pred)
        processed += 1
        if index % log_interval == 0 or index == total_series:
            logger.info(
                msg=(
                    "Seq2Seq forecasting progress: "
                    f"{index}/{total_series} series (skipped={skipped})."
                )
            )
    logger.info(msg=(f"Seq2Seq forecasting complete: processed={processed}, skipped={skipped}."))
    return pd.concat(objs=predictions, ignore_index=True)


class Seq2SeqForecaster(ForecastModel):
    """Seq2Seq forecaster with optional DDP training."""

    def __init__(self, config: Seq2SeqConfig | None = None) -> None:
        if config is None:
            config = Seq2SeqConfig(
                context_length=90,
                hidden_size=64,
                epochs=5,
                batch_size=64,
                learning_rate=1e-3,
            )
        self._config = config

    @property
    def name(self) -> str:
        return "Seq2Seq"

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        freq: str,
    ) -> ModelResult:
        """
        Train the Seq2Seq model and forecast future values.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training data.
        horizon : int
            Forecast horizon.
        freq : str
            Pandas frequency string.

        Returns
        -------
        ModelResult
            Forecasting result.
        """
        logger = _get_experiment_logger()
        start_time = time.perf_counter()
        series_values = build_series_values(train_df=train_df)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        logger.info(
            msg=(
                "Seq2Seq fit_predict start: "
                f"series={len(series_values)}, horizon={horizon}, device={device_type}."
            )
        )
        with NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            model_state_path = Path(temp_file.name)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(msg="Seq2Seq using distributed training.")
            model = train_seq2seq_distributed(
                series_values=series_values,
                config=self._config,
                horizon=horizon,
                model_state_path=model_state_path,
            )
            model_state_path.unlink(missing_ok=True)
        else:
            logger.info(msg="Seq2Seq using single-GPU/CPU training.")
            model = train_seq2seq_single(
                series_values=series_values,
                config=self._config,
                horizon=horizon,
                device=device,
            )
        pred_df = forecast_seq2seq(
            model=model,
            train_df=train_df,
            context_length=self._config.context_length,
            horizon=horizon,
            freq=freq,
            device=device,
        )
        elapsed = time.perf_counter() - start_time
        logger.info(msg=f"Seq2Seq fit_predict complete in {elapsed:.2f}s.")
        return ModelResult(
            model_name=self.name,
            predictions=pred_df,
            train_time_sec=elapsed,
        )
