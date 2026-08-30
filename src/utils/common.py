"""
src/utils/common.py
---------------------------------------------------------------------------
Shared utilities: config loading, reproducibility seeding, GPU detection,
and logging setup used across the whole pipeline.
---------------------------------------------------------------------------
"""
import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the central YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml, relative to the project root or absolute.

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / config_path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and (if available) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return a torch.device, automatically using CUDA if available, else CPU."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    return device


def setup_logger(name: str = "flood_pipeline", log_file: str = None) -> logging.Logger:
    """Configure and return a logger that writes to console (and optionally a file)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def ensure_dirs(*dirs) -> None:
    """Create directories (and parents) if they do not already exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
