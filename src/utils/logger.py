"""
Centralized structured logger for the JWST MIRI IFU Spectral Pipeline.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "jwst_ifu_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Retrieve or configure a structured logger instance.

    Parameters
    ----------
    name : str
        Logger module identifier.
    level : int
        Logging verbosity level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.formatters.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ) if hasattr(logging, "formatters") else logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
