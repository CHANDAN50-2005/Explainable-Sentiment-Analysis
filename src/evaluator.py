"""
src/evaluator.py
Evaluation utilities: computes metrics and generates report charts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

from src.utils import get_logger, save_json, ensure_dir

logger = get_logger(__name__)

LABEL_NAMES = ["NEGATIVE", "POSITIVE"]


# ─── Prediction collection ────────────────────────────────────────────────────

def collect_predictions(
    model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Run the model over all batches.

    Returns:
        y_true, y_pred, y_prob_positive
    """
    model.model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].tolist()

            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1).tolist()

            y_true.extend(labels)
            y_pred.extend(preds)
            y_prob.extend(probs[:, 1].tolist())  # probability of POSITIVE

    return y_true, y_pred, y_prob


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
) -> Dict:
    """Compute and return all classification metrics as a dict."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None

    report = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, output_dict=True
    )

    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "roc_auc": round(auc, 4) if auc else None,
        "classification_report": report,
    }
    logger.info("Accuracy: %.4f | F1 (macro): %.4f | AUC: %s", acc, f1, auc)
    return metrics


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    report_dir: str,
) -> str:
    """Save a seaborn confusion matrix heatmap. Returns file path."""
    ensure_dir(report_dir)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved: %s", path)
    return path


def plot_roc_curve(
    y_true: List[int],
    y_prob: List[float],
    report_dir: str,
) -> str:
    """Save an ROC curve plot. Returns file path."""
    ensure_dir(report_dir)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4f8ef7", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    path = os.path.join(report_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("ROC curve saved: %s", path)
    return path


def plot_training_history(history: Dict, report_dir: str) -> str:
    """Plot training loss and validation accuracy curves. Returns file path."""
    ensure_dir(report_dir)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss subplot
    axes[0].plot(epochs, history["train_loss"], "o-", color="#f97316", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "s-", color="#4f8ef7", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy subplot
    axes[1].plot(epochs, history["val_accuracy"], "D-", color="#10b981", label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy", fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(report_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Training history saved: %s", path)
    return path


# ─── Full evaluation runner ──────────────────────────────────────────────────

def run_full_evaluation(
    model,
    val_loader: DataLoader,
    report_dir: str,
) -> Dict:
    """
    Collect predictions, compute all metrics, generate all charts.
    Returns the metrics dict.
    """
    logger.info("Running full evaluation …")
    y_true, y_pred, y_prob = collect_predictions(model, val_loader, model.device)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    plot_confusion_matrix(y_true, y_pred, report_dir)
    plot_roc_curve(y_true, y_prob, report_dir)

    save_json(metrics, os.path.join(report_dir, "metrics.json"))
    logger.info("Evaluation complete. Reports saved to: %s", report_dir)
    return metrics
