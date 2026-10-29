# 🧠 Explainable Transformer-Based Sentiment Analysis System

> An end-to-end NLP project that classifies text sentiment using **DistilBERT** and explains every prediction with **LIME** word importance scores and **attention weight heatmaps** — all served through a sleek interactive web dashboard.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Model** | Fine-tuned `distilbert-base-uncased` on SST-2 |
| **Accuracy** | ~92–94% on SST-2 validation set |
| **Explainability** | LIME + Transformer attention weights |
| **API** | FastAPI REST endpoints |
| **Dashboard** | Dark-mode interactive web UI |
| **Metrics** | Accuracy, F1, AUC-ROC, Confusion Matrix |

---

## 🗂️ Project Structure

```
NLP_MiniProject/
├── data/                    # Datasets (auto-downloaded)
├── models/                  # Saved model checkpoints
├── notebooks/               # Jupyter exploration notebooks
├── src/
│   ├── data_loader.py       # Data pipeline & tokenization
│   ├── model.py             # DistilBERT model & training loop
│   ├── evaluator.py         # Metrics & chart generation
│   ├── explainer.py         # LIME + attention explainability
│   └── utils.py             # Shared utilities
├── api/
│   ├── main.py              # FastAPI application
│   └── schemas.py           # Pydantic models
├── frontend/
│   ├── index.html           # Dashboard UI
│   ├── style.css            # Dark-mode styles
│   └── app.js               # Dashboard logic
├── reports/                 # Auto-generated evaluation charts
├── config.yaml              # All project configuration
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Downloads SST-2 dataset and fine-tunes DistilBERT (~20–40 min on CPU, ~5 min on GPU):

```bash
python train.py
```

### 3. Evaluate & Generate Reports

```bash
python evaluate.py
```

Generates:
- `reports/metrics.json` — All metrics
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/training_history.png`
- `reports/attention_examples/` — Example heatmaps

### 4. Start the API Server

```bash
python -m uvicorn api.main:app --reload --port 8000
```

### 5. Open the Dashboard

Visit **http://localhost:8000** in your browser.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard |
| `GET` | `/health` | Model status check |
| `POST` | `/predict` | Sentiment prediction |
| `POST` | `/explain` | Prediction + LIME + attention |
| `GET` | `/metrics` | Model evaluation metrics |
| `GET` | `/examples` | Demo sentences |
| `GET` | `/docs` | Interactive API docs (Swagger) |

### Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The movie was absolutely brilliant!"}'
```

Response:
```json
{
  "label": "POSITIVE",
  "label_id": 1,
  "confidence": 0.9823,
  "probabilities": {
    "NEGATIVE": 0.0177,
    "POSITIVE": 0.9823
  },
  "processing_time_ms": 45.3
}
```

---

## ⚙️ Configuration

All settings are in [`config.yaml`](config.yaml):

```yaml
model:
  base_model: "distilbert-base-uncased"
  max_length: 128

training:
  epochs: 3
  learning_rate: 2.0e-5
  batch_size: 16

explainability:
  lime_num_samples: 500    # Higher = more accurate but slower
  lime_num_features: 10    # Top N words to explain
```

**For quick testing**, set in `config.yaml`:
```yaml
data:
  max_train_samples: 2000   # use only 2000 training examples
  max_val_samples: 500
training:
  epochs: 1
```

---

## 📊 Expected Results

| Metric | Value |
|---|---|
| Accuracy | ~92–94% |
| F1 Score (macro) | ~0.91–0.93 |
| AUC-ROC | ~0.97–0.99 |
| Inference time | <100ms/sample |

---

## 🧪 How Explainability Works

### LIME (Local Interpretable Model-Agnostic Explanations)
1. Takes the input sentence
2. Creates 500 perturbed versions (randomly masking words)
3. Measures how the prediction changes for each version
4. Assigns importance scores to each word based on its impact

### Attention Weights
1. Extracts attention matrices from all 6 DistilBERT layers
2. Averages across all 12 attention heads
3. Uses the `[CLS]` token's attention to other tokens
4. Normalizes scores to [0, 1] for visualization

---

## 🛠️ Tech Stack

- **Transformers**: HuggingFace `transformers` library
- **Model**: `distilbert-base-uncased` (66M parameters)
- **Dataset**: Stanford Sentiment Treebank v2 (SST-2)
- **Explainability**: `lime` library
- **Backend**: `FastAPI` + `uvicorn`
- **Frontend**: Vanilla HTML/CSS/JS
- **ML utilities**: `scikit-learn`, `matplotlib`, `seaborn`

---

## 📄 License

MIT License — free for educational and research use.
