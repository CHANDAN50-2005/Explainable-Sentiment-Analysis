"""
api/main.py
FastAPI application — serves predictions, explanations, and the frontend.

Run with:
    python -m uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    TextRequest,
)
from src.explainer import SentimentExplainer
from src.model import SentimentModel
from src.utils import load_config, load_json, get_logger
from transformers import DistilBertTokenizerFast

logger = get_logger("api")

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Explainable Sentiment Analysis API",
    description=(
        "A transformer-based sentiment analysis API with LIME and attention "
        "explainability. Built with DistilBERT + FastAPI."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve the frontend as static files ────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ─── Global model state ───────────────────────────────────────────────────────

_model: SentimentModel | None = None
_tokenizer: DistilBertTokenizerFast | None = None
_explainer: SentimentExplainer | None = None
_config: dict | None = None


def _load_model():
    """Load model and tokenizer on first request (lazy loading)."""
    global _model, _tokenizer, _explainer, _config

    if _model is not None:
        return  # already loaded

    _config = load_config("config.yaml")
    save_dir = _config["model"]["save_dir"]

    if not Path(save_dir).exists():
        raise RuntimeError(
            f"No trained model found at '{save_dir}'. "
            "Please run  python train.py  first."
        )

    logger.info("Loading model from %s …", save_dir)
    _model = SentimentModel.load(_config)
    _tokenizer = DistilBertTokenizerFast.from_pretrained(save_dir)
    _explainer = SentimentExplainer(_model, _tokenizer, _config)
    logger.info("Model loaded ✅")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, summary="Serve the frontend dashboard")
async def root():
    """Serve the main dashboard HTML."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(
        "<h2>Frontend not found. Open <code>frontend/index.html</code> directly.</h2>"
    )


@app.get("/health", response_model=HealthResponse, summary="API health check")
async def health():
    """Check if the API is running and whether the model is loaded."""
    config = load_config("config.yaml")
    model_loaded = Path(config["model"]["save_dir"]).exists()
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_name=config["model"]["base_model"],
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict sentiment")
async def predict(request: TextRequest):
    """
    Classify the sentiment of input text.

    Returns:
    - **label**: POSITIVE or NEGATIVE
    - **confidence**: model's confidence score (0–1)
    - **probabilities**: softmax probabilities for each class
    """
    try:
        _load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    t0 = time.perf_counter()
    result = _model.predict(request.text, _tokenizer, _config["model"]["max_length"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        text=request.text,
        label=result["label"],
        label_id=result["label_id"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/explain", summary="Explain sentiment prediction")
async def explain(request: TextRequest):
    """
    Classify sentiment AND explain the prediction with:
    - **LIME word importance scores** (which words drove the decision)
    - **Attention weights** (which tokens the model focused on)
    """
    try:
        _load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    t0 = time.perf_counter()
    result = _explainer.explain(request.text)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return JSONResponse(content={
        "text": result["text"],
        "prediction": {
            "label": result["prediction"]["label"],
            "confidence": result["prediction"]["confidence"],
            "probabilities": result["prediction"]["probabilities"],
            "processing_time_ms": round(elapsed_ms, 2),
        },
        "lime": {
            "word_scores": result["lime"]["word_scores"],
            "positive_words": result["lime"]["positive_words"],
            "negative_words": result["lime"]["negative_words"],
        },
        "attention": {
            "tokens": result["attention"]["tokens"],
            "scores": result["attention"]["scores"],
        },
        "processing_time_ms": round(elapsed_ms, 2),
    })


@app.get("/metrics", response_model=MetricsResponse, summary="Get model evaluation metrics")
async def get_metrics():
    """Return pre-computed evaluation metrics from the reports folder."""
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        return MetricsResponse(
            available=False,
            message="Metrics not found. Run  python evaluate.py  first.",
        )
    data = load_json(str(metrics_path))
    return MetricsResponse(
        accuracy=data.get("accuracy"),
        f1_macro=data.get("f1_macro"),
        f1_weighted=data.get("f1_weighted"),
        precision_macro=data.get("precision_macro"),
        recall_macro=data.get("recall_macro"),
        roc_auc=data.get("roc_auc"),
        available=True,
    )


@app.get("/examples", summary="Get example sentences for demo")
async def get_examples():
    """Return a set of example sentences for testing the dashboard."""
    return {
        "examples": [
            {"text": "The movie was absolutely brilliant and captivating from start to finish!", "expected": "POSITIVE"},
            {"text": "Terrible acting and a boring plot. I fell asleep halfway through.", "expected": "NEGATIVE"},
            {"text": "One of the best films I have ever seen. Highly recommended!", "expected": "POSITIVE"},
            {"text": "I completely hated this film. What a waste of time and money.", "expected": "NEGATIVE"},
            {"text": "The special effects were stunning and the soundtrack was phenomenal.", "expected": "POSITIVE"},
            {"text": "Absolutely dreadful. The script was painful to listen to.", "expected": "NEGATIVE"},
            {"text": "A masterpiece of modern cinema. Every scene was breathtaking.", "expected": "POSITIVE"},
            {"text": "Disappointing and predictable. Nothing new or interesting.", "expected": "NEGATIVE"},
        ]
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = load_config("config.yaml")
    uvicorn.run(
        "api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=cfg["api"]["reload"],
    )
