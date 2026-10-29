"""
train.py
Main training entry point.

Usage:
    python train.py
    python train.py --config config.yaml
"""
import argparse
import sys
import os

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, set_seed, get_logger
from src.data_loader import get_dataloaders
from src.model import SentimentModel
from src.evaluator import plot_training_history

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DistilBERT sentiment model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    logger.info("=" * 60)
    logger.info("  Explainable Transformer Sentiment Analysis — TRAINING")
    logger.info("=" * 60)
    logger.info("Model  : %s", config["model"]["base_model"])
    logger.info("Epochs : %d", config["training"]["epochs"])
    logger.info("LR     : %s", config["training"]["learning_rate"])
    logger.info("Device : will be detected automatically")
    logger.info("=" * 60)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    logger.info("Step 1/3 — Loading data …")
    train_loader, val_loader, tokenizer = get_dataloaders(config)

    # ── Step 2: Train model ──────────────────────────────────────────────────
    logger.info("Step 2/3 — Fine-tuning DistilBERT …")
    model = SentimentModel(config)
    history = model.train(train_loader, val_loader, tokenizer)

    # ── Step 3: Plot training curves ─────────────────────────────────────────
    logger.info("Step 3/3 — Saving training history plot …")
    plot_training_history(history, config["evaluation"]["report_dir"])

    logger.info("✅ Training complete!")
    logger.info("   Model saved  → %s", config["model"]["save_dir"])
    logger.info("   History plot → %s/training_history.png", config["evaluation"]["report_dir"])
    logger.info("")
    logger.info("Next step: Run  python evaluate.py  to generate full evaluation report.")


if __name__ == "__main__":
    main()
