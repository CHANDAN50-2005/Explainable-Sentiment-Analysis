"""
smoke_test.py — Quick validation that all modules import and basic logic works.
"""
import sys
sys.path.insert(0, ".")

from src.utils import load_config, get_device, set_seed, get_logger
from src.data_loader import clean_text
from src.evaluator import compute_metrics

logger = get_logger("smoke_test")
set_seed(42)
device = get_device()
logger.info("Device: %s", device)

# ── Text cleaning check ────────────────────────────────────────────────────────
cleaned = clean_text("<b>Hello</b>  World!   https://example.com")
assert cleaned == "Hello World! [URL]", f"Got: {repr(cleaned)}"
logger.info("clean_text: OK → %s", repr(cleaned))

# ── Config check ───────────────────────────────────────────────────────────────
cfg = load_config("config.yaml")
assert cfg["model"]["base_model"] == "distilbert-base-uncased"
logger.info("config.yaml: OK → model=%s", cfg["model"]["base_model"])

# ── Metrics check ─────────────────────────────────────────────────────────────
m = compute_metrics([0, 1, 0, 1], [0, 1, 1, 1], [0.1, 0.9, 0.6, 0.8])
assert "accuracy" in m and "f1_macro" in m
logger.info("compute_metrics: OK → accuracy=%s", m["accuracy"])

print()
print("=" * 45)
print("  [PASSED]  Smoke test OK - all modules work!")
print("  Device   : " + str(device))
print("  Model    : " + cfg["model"]["base_model"])
print("  Accuracy : " + str(m["accuracy"]))
print("=" * 45)
