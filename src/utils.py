"""
src/utils.py
Shared utility functions used across the project.
"""
import os
import json
import random
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# ─── Logger ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load the project YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── File I/O ─────────────────────────────────────────────────────────────────

def save_json(data: Any, path: str) -> None:
    """Save data as a JSON file, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load JSON from a file."""
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ─── Label mapping helpers ────────────────────────────────────────────────────

ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}


def id_to_label(idx: int) -> str:
    return ID2LABEL.get(idx, "UNKNOWN")


def label_to_id(label: str) -> int:
    return LABEL2ID.get(label.upper(), -1)
