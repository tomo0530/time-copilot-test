import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal


def get_custom_logger(
    log_dir: str = "./",
    log_file_name: str = "app.log",
    logger_name: str = "custom_logger",
    max_bytes: int = 1_000_000,
    backup_count: int = 5,
    level_file: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
    level_stdout: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> logging.Logger:
    """
    Configure and return a standard logging logger with stdout and rotating file handlers.

    Parameters
    ----------
    log_dir : str
        Directory to store log files.
    log_file_name : str
        Name of the log file.
    logger_name : str
        Name of the logger instance. Used to retrieve the logger by name.
    max_bytes : int
        Max size in bytes before rotating the log file.
    backup_count : int
        Number of rotated files to retain.
    level_file : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        Log level for file output.
    level_stdout : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        Log level for stdout.

    Returns
    -------
    logging.Logger
        Configured logger instance with stdout and rotating file handlers.

    Notes
    -----
    This function modifies the global logger state for the given `logger_name`.
    The base logger level is set to DEBUG to allow filtering by individual handlers.
    """

    log_path = Path(log_dir).expanduser()
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(
        logging.DEBUG
    )  # Set the base logger level to DEBUG; filtering is done by handlers.

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatters for stdout and file
    stdout_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler for stdout logging
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level_stdout)
    stdout_handler.setFormatter(stdout_formatter)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_path / log_file_name,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level_file)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    return logger
