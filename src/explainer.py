"""
src/explainer.py
Explainability module: LIME text explanations + attention weight extraction.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import DistilBertTokenizerFast

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


# ─── LIME Explainer ───────────────────────────────────────────────────────────

class SentimentExplainer:
    """
    Provides two explanation methods:
      1. LIME – word-level importance via perturbation
      2. Attention – token importance from DistilBERT attention weights
    """

    def __init__(
        self,
        model,                          # SentimentModel instance
        tokenizer: DistilBertTokenizerFast,
        config: Dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.exp_cfg = config["explainability"]
        self.model_cfg = config["model"]
        self.device = model.device

        # labels keys may be int (YAML) or str — normalize to int
        _labels = {int(k): v for k, v in self.model_cfg["labels"].items()}
        self.label_names = [
            _labels[0],
            _labels[1],
        ]

        self.lime_explainer = LimeTextExplainer(
            class_names=self.label_names,
            random_state=42,
        )

    # ── LIME ──────────────────────────────────────────────────────────────────

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Batch prediction function consumed by LIME."""
        self.model.model.eval()
        all_probs = []

        max_len = self.model_cfg["max_length"]
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)

        return np.array(all_probs)

    def explain_lime(self, text: str) -> Dict:
        """
        Run LIME on a single text.

        Returns a dict with:
          - word_scores: list of (word, score) tuples
          - label: predicted label
          - positive_words: top words pushing toward POSITIVE
          - negative_words: top words pushing toward NEGATIVE
        """
        prediction = self.model.predict(text, self.tokenizer, self.model_cfg["max_length"])
        label_id = prediction["label_id"]

        logger.info("Running LIME explanation (n_samples=%d)…", self.exp_cfg["lime_num_samples"])
        exp = self.lime_explainer.explain_instance(
            text,
            self.predict_proba_wrapper,
            num_features=self.exp_cfg["lime_num_features"],
            num_samples=self.exp_cfg["lime_num_samples"],
            labels=[0, 1],
        )

        # Scores for predicted class
        word_scores = exp.as_list(label=label_id)

        positive_words = [(w, s) for w, s in word_scores if s > 0]
        negative_words = [(w, s) for w, s in word_scores if s < 0]

        return {
            "word_scores": word_scores,
            "positive_words": positive_words,
            "negative_words": negative_words,
            "predicted_label": prediction["label"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
        }

    def predict_proba_wrapper(self, texts: List[str]) -> np.ndarray:
        """Public wrapper for LIME compatibility."""
        return self._predict_proba(texts)

    # ── Attention ─────────────────────────────────────────────────────────────

    def get_attention_scores(self, text: str) -> Dict:
        """
        Extract per-token attention scores from the model.

        Averages attention across all heads in the last layer, then over
        the [CLS] token's outgoing attention → gives each token a scalar.

        Returns:
          - tokens: list of visible tokens (no padding)
          - scores: corresponding attention scores (normalized)
          - raw_text: original text
        """
        prediction = self.model.predict(text, self.tokenizer, self.model_cfg["max_length"])
        attentions = prediction["attentions"]   # tuple (num_layers,) of (1, heads, seq, seq)
        tokens = prediction["tokens"]
        attention_mask = prediction["attention_mask"][0].tolist()

        # Last layer attention: shape (1, num_heads, seq_len, seq_len)
        last_layer = attentions[-1].squeeze(0)   # (num_heads, seq_len, seq_len)

        # Mean across heads → (seq_len, seq_len)
        avg_attention = last_layer.mean(dim=0).cpu().numpy()

        # CLS token [0] attends to other tokens
        cls_attention = avg_attention[0]

        # Find real token count (remove padding)
        real_len = sum(attention_mask)
        real_tokens = tokens[:real_len]
        real_scores = cls_attention[:real_len]

        # Normalize to [0, 1]
        real_scores = real_scores / (real_scores.max() + 1e-8)

        # Remove special tokens [CLS] and [SEP] for display
        display_tokens = []
        display_scores = []
        for tok, score in zip(real_tokens, real_scores):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            display_tokens.append(tok.replace("##", ""))  # merge WordPiece
            display_scores.append(float(score))

        return {
            "tokens": display_tokens,
            "scores": display_scores,
            "predicted_label": prediction["label"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "raw_text": text,
        }

    # ── Visualization helpers ─────────────────────────────────────────────────

    def plot_lime_chart(
        self,
        text: str,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate LIME bar chart. Returns saved image path."""
        result = self.explain_lime(text)
        word_scores = sorted(result["word_scores"], key=lambda x: x[1])

        words = [w for w, _ in word_scores]
        scores = [s for _, s in word_scores]
        colors = ["#ef4444" if s < 0 else "#10b981" for s in scores]

        fig, ax = plt.subplots(figsize=(8, max(4, len(words) * 0.5 + 1)))
        bars = ax.barh(words, scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_xlabel("LIME Importance Score", fontsize=11)
        ax.set_title(
            f"Word Importance — Predicted: {result['predicted_label']} "
            f"({result['confidence']*100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_facecolor("#1e293b")
        fig.patch.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        plt.tight_layout()

        if save_path is None:
            ensure_dir("reports/attention_examples")
            save_path = "reports/attention_examples/lime_chart.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_attention_heatmap(
        self,
        text: str,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate attention score heatmap over tokens. Returns saved image path."""
        result = self.get_attention_scores(text)
        tokens = result["tokens"]
        scores = np.array(result["scores"])

        # Chunk tokens into rows of max 12 for readability
        chunk_size = 12
        rows = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        score_rows = [scores[i : i + chunk_size] for i in range(0, len(scores), chunk_size)]

        fig, axes = plt.subplots(len(rows), 1, figsize=(14, max(2, len(rows) * 1.8)))
        if len(rows) == 1:
            axes = [axes]

        fig.patch.set_facecolor("#0f172a")

        for ax, row_tokens, row_scores in zip(axes, rows, score_rows):
            # Pad row to chunk_size for uniform heatmap
            pad = chunk_size - len(row_tokens)
            row_scores_padded = np.pad(row_scores, (0, pad), constant_values=np.nan)
            row_tokens_padded = row_tokens + [""] * pad

            data = row_scores_padded.reshape(1, -1)
            im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

            ax.set_xticks(range(chunk_size))
            ax.set_xticklabels(row_tokens_padded, fontsize=9, color="white")
            ax.set_yticks([])
            ax.set_facecolor("#0f172a")

            for j, (tok, sc) in enumerate(zip(row_tokens, row_scores)):
                text_color = "black" if sc > 0.5 else "white"
                ax.text(j, 0, f"{sc:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

        plt.suptitle(
            f"Attention Heatmap — Predicted: {result['predicted_label']} "
            f"({result['confidence']*100:.1f}%)",
            fontsize=12, fontweight="bold", color="white",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path is None:
            ensure_dir("reports/attention_examples")
            save_path = "reports/attention_examples/attention_heatmap.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    # ── Combined explain ──────────────────────────────────────────────────────

    def explain(self, text: str) -> Dict:
        """
        Run both LIME and Attention explanation on a text.
        Returns a combined dict suitable for the API response.
        """
        lime_result = self.explain_lime(text)
        attention_result = self.get_attention_scores(text)

        return {
            "text": text,
            "prediction": {
                "label": lime_result["predicted_label"],
                "confidence": lime_result["confidence"],
                "probabilities": lime_result["probabilities"],
            },
            "lime": {
                "word_scores": lime_result["word_scores"],
                "positive_words": lime_result["positive_words"],
                "negative_words": lime_result["negative_words"],
            },
            "attention": {
                "tokens": attention_result["tokens"],
                "scores": attention_result["scores"],
            },
        }
