"""
src/model.py
DistilBERT fine-tuning model definition and training loop.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from src.utils import get_logger, get_device, set_seed, save_json

logger = get_logger(__name__)


# ─── Model wrapper ────────────────────────────────────────────────────────────

class SentimentModel:
    """
    Wraps DistilBertForSequenceClassification with training, saving,
    loading, and single-text inference helpers.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_cfg = config["model"]
        self.train_cfg = config["training"]
        self.device = get_device()

        logger.info("Device: %s", self.device)

        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_cfg["base_model"],
            num_labels=self.model_cfg["num_labels"],
            id2label={int(k): v for k, v in self.model_cfg["labels"].items()},
            label2id={v: int(k) for k, v in self.model_cfg["labels"].items()},
            dropout=self.model_cfg["dropout"],               # DistilBERT general dropout
            attention_dropout=self.model_cfg["dropout"],     # DistilBERT attention dropout
            seq_classif_dropout=self.model_cfg["dropout"],   # classifier head dropout
            attn_implementation="eager",                     # required for output_attentions=True
        )
        self.model.to(self.device)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: DistilBertTokenizerFast,
    ) -> Dict:
        """Full fine-tuning loop. Returns history dict with loss/accuracy."""
        set_seed(self.train_cfg["seed"])

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        total_steps = len(train_loader) * self.train_cfg["epochs"]
        warmup_steps = int(total_steps * self.train_cfg["warmup_ratio"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_acc = 0.0

        for epoch in range(1, self.train_cfg["epochs"] + 1):
            # ── Train phase ───────────────────────────────────────────────────
            self.model.train()
            total_train_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{self.train_cfg['epochs']} [Train]")

            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)

            # ── Validation phase ──────────────────────────────────────────────
            val_loss, val_acc = self._evaluate(val_loader)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            logger.info(
                "Epoch %d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f",
                epoch, avg_train_loss, val_loss, val_acc,
            )

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save(tokenizer)
                logger.info("✅ New best model saved (val_acc=%.4f)", best_val_acc)

        logger.info("Training complete. Best Val Accuracy: %.4f", best_val_acc)
        save_json(history, os.path.join(self.model_cfg["save_dir"], "training_history.json"))
        return history

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Run evaluation on a DataLoader, return (avg_loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="  Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(loader), correct / total

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 128,
    ) -> Dict:
        """
        Predict sentiment for a single text string.

        Returns a dict with keys: label, label_id, confidence, probabilities.
        Also returns the model outputs for attention extraction.
        """
        self.model.eval()
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        label_id = int(torch.argmax(outputs.logits, dim=-1).item())
        label_map = {int(k): v for k, v in self.model_cfg["labels"].items()}

        return {
            "label": label_map[label_id],
            "label_id": label_id,
            "confidence": round(probs[label_id], 4),
            "probabilities": {
                label_map[0]: round(probs[0], 4),
                label_map[1]: round(probs[1], 4),
            },
            "attentions": outputs.attentions,          # tuple of tensors
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, tokenizer: DistilBertTokenizerFast) -> None:
        """Save model and tokenizer to the configured save directory."""
        save_dir = self.model_cfg["save_dir"]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info("Model saved to: %s", save_dir)

    @classmethod
    def load(cls, config: Dict) -> "SentimentModel":
        """Load a fine-tuned model from the save directory."""
        instance = cls.__new__(cls)
        instance.config = config
        instance.model_cfg = config["model"]
        instance.train_cfg = config["training"]
        instance.device = get_device()

        save_dir = config["model"]["save_dir"]
        logger.info("Loading model from: %s", save_dir)
        instance.model = DistilBertForSequenceClassification.from_pretrained(
            save_dir,
            attn_implementation="eager",   # required for output_attentions=True in /explain
        )
        instance.model.to(instance.device)
        instance.model.eval()
        return instance
