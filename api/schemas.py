"""
api/schemas.py
Pydantic v2 request/response models for the FastAPI endpoints.
"""
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ─── Request models ───────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Input text to analyze",
        examples=["The movie was absolutely fantastic!"],
    )


# ─── Response models ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    text: str
    label: str                          # "POSITIVE" or "NEGATIVE"
    label_id: int
    confidence: float
    probabilities: Dict[str, float]     # {"NEGATIVE": 0.12, "POSITIVE": 0.88}
    processing_time_ms: float


class LimeWord(BaseModel):
    word: str
    score: float


class AttentionData(BaseModel):
    tokens: List[str]
    scores: List[float]


class ExplainResponse(BaseModel):
    text: str
    prediction: PredictionResponse
    lime: Dict                          # word_scores, positive_words, negative_words
    attention: AttentionData
    processing_time_ms: float


class MetricsResponse(BaseModel):
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    roc_auc: Optional[float] = None
    available: bool = True
    message: str = "Metrics loaded successfully"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    version: str = "1.0.0"
