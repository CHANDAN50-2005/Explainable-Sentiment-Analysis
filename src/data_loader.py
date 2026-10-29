"""
src/data_loader.py
Data pipeline: loads SST-2, preprocesses, tokenizes, and returns DataLoaders.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast

from src.utils import get_logger, load_config

logger = get_logger(__name__)


# ─── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning: strip HTML, normalize whitespace."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── HuggingFace Dataset wrapper ─────────────────────────────────────────────

class SentimentDataset(Dataset):
    """Wraps a HuggingFace dataset split into a PyTorch Dataset."""

    def __init__(self, hf_dataset, tokenizer: DistilBertTokenizerFast, max_length: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        text = clean_text(example["sentence"])
        label = int(example["label"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ─── Main loader function ─────────────────────────────────────────────────────

def get_dataloaders(
    config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DistilBertTokenizerFast]:
    """
    Load and tokenize the SST-2 dataset and return train/val DataLoaders.

    Returns:
        train_loader, val_loader, tokenizer
    """
    if config is None:
        config = load_config()

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    logger.info("Loading tokenizer: %s", model_cfg["base_model"])
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_cfg["base_model"])

    logger.info("Downloading dataset: %s", data_cfg["dataset_name"])
    raw_datasets: DatasetDict = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config", "default"),
    )

    train_split = raw_datasets[data_cfg["train_split"]]
    val_split = raw_datasets[data_cfg["validation_split"]]

    # Optional: limit samples for quick local testing
    if data_cfg.get("max_train_samples"):
        train_split = train_split.select(range(data_cfg["max_train_samples"]))
        logger.info("Limiting train to %d samples", data_cfg["max_train_samples"])
    if data_cfg.get("max_val_samples"):
        val_split = val_split.select(range(data_cfg["max_val_samples"]))

    logger.info("Train: %d samples | Val: %d samples", len(train_split), len(val_split))

    max_len = model_cfg["max_length"]
    train_dataset = SentimentDataset(train_split, tokenizer, max_len)
    val_dataset = SentimentDataset(val_split, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, tokenizer


def get_raw_dataset(config: Optional[Dict] = None) -> DatasetDict:
    """Return the raw HuggingFace DatasetDict (for notebooks)."""
    if config is None:
        config = load_config()
    return load_dataset(
        config["data"]["dataset_name"],
        config["data"].get("dataset_config", "default"),
    )
