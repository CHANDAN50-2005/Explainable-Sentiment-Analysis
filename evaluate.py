"""
evaluate.py
Evaluation entry point — generates metrics, confusion matrix, ROC curve.

Usage:
    python evaluate.py
    python evaluate.py --config config.yaml
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, get_logger
from src.data_loader import get_dataloaders
from src.model import SentimentModel
from src.evaluator import run_full_evaluation
from src.explainer import SentimentExplainer
from transformers import DistilBertTokenizerFast

logger = get_logger("evaluate")

DEMO_TEXTS = [
    "The movie was absolutely brilliant and captivating from start to finish.",
    "Terrible acting and a boring plot. I fell asleep halfway through.",
    "The special effects were amazing but the story was quite predictable.",
    "I completely hated this film. What a waste of time.",
    "One of the best films I have ever seen. Highly recommended!",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-lime", action="store_true", help="Skip LIME (faster)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    report_dir = config["evaluation"]["report_dir"]

    logger.info("=" * 60)
    logger.info("  Explainable Transformer Sentiment Analysis — EVALUATION")
    logger.info("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info("Loading fine-tuned model from: %s", config["model"]["save_dir"])
    model = SentimentModel.load(config)
    tokenizer = DistilBertTokenizerFast.from_pretrained(config["model"]["save_dir"])

    # ── Load val data ────────────────────────────────────────────────────────
    logger.info("Loading validation data …")
    _, val_loader, _ = get_dataloaders(config)

    # ── Run evaluation ───────────────────────────────────────────────────────
    logger.info("Computing metrics …")
    metrics = run_full_evaluation(model, val_loader, report_dir)

    print("\n" + "=" * 50)
    print("  [RESULTS] EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 (macro)      : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted)   : {metrics['f1_weighted']:.4f}")
    print(f"  Precision       : {metrics['precision_macro']:.4f}")
    print(f"  Recall          : {metrics['recall_macro']:.4f}")
    print(f"  ROC AUC         : {metrics['roc_auc']}")
    print("=" * 50 + "\n")

    # ── Demo predictions ─────────────────────────────────────────────────────
    print("  [DEMO] DEMO PREDICTIONS")
    print("=" * 50)
    for text in DEMO_TEXTS:
        result = model.predict(text, tokenizer, config["model"]["max_length"])
        bar = "[+]" if result["label"] == "POSITIVE" else "[-]"
        print(f"  {bar} [{result['label']:8s}] ({result['confidence']*100:.1f}%) -- {text[:65]}...")
    print()

    # ── Generate explainability examples ────────────────────────────────────
    if not args.skip_lime:
        logger.info("Generating explainability examples …")
        explainer = SentimentExplainer(model, tokenizer, config)
        attn_dir = config["evaluation"]["attention_examples_dir"]

        for i, text in enumerate(DEMO_TEXTS[:3]):
            logger.info("  Example %d: %s", i + 1, text[:50])
            explainer.plot_attention_heatmap(
                text, save_path=os.path.join(attn_dir, f"attention_{i+1}.png")
            )
            explainer.plot_lime_chart(
                text, save_path=os.path.join(attn_dir, f"lime_{i+1}.png")
            )

    logger.info("✅ All reports saved to: %s/", report_dir)
    logger.info("Next step: Run  python -m uvicorn api.main:app --reload  to start the API.")


if __name__ == "__main__":
    main()
