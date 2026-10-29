/**
 * app.js — SentimentAI dashboard logic
 * Handles: API calls, prediction rendering, attention heatmap,
 *          LIME bar chart, confidence gauge, examples modal, status check.
 */

const API_BASE = window.location.origin;

// ── DOM refs ────────────────────────────────────────────────────────────────
const textInput       = document.getElementById('textInput');
const btnAnalyze      = document.getElementById('btnAnalyze');
const btnClear        = document.getElementById('btnClear');
const btnExamples     = document.getElementById('btnExamples');
const charCount       = document.getElementById('charCount');
const resultsContainer= document.getElementById('resultsContainer');
const errorToast      = document.getElementById('errorToast');
const toastMsg        = document.getElementById('toastMsg');
const examplesModal   = document.getElementById('examplesModal');
const modalClose      = document.getElementById('modalClose');
const exampleList     = document.getElementById('exampleList');
const statusDot       = document.getElementById('statusDot');
const statusText      = document.getElementById('statusText');

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadMetrics();
  loadExamples();
});

// ── Input events ────────────────────────────────────────────────────────────
textInput.addEventListener('input', () => {
  const len = textInput.value.length;
  charCount.textContent = `${len} / 2000`;
  btnAnalyze.disabled = len < 2;
  charCount.style.color = len > 1800 ? '#f97316' : '';
});

textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) btnAnalyze.click();
});

btnClear.addEventListener('click', () => {
  textInput.value = '';
  textInput.dispatchEvent(new Event('input'));
  resultsContainer.hidden = true;
});

btnAnalyze.addEventListener('click', analyze);
btnExamples.addEventListener('click', () => { examplesModal.hidden = false; });
modalClose.addEventListener('click', () => { examplesModal.hidden = true; });
examplesModal.addEventListener('click', (e) => {
  if (e.target === examplesModal) examplesModal.hidden = true;
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !examplesModal.hidden) examplesModal.hidden = true;
});

// ── Health check ────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.model_loaded) {
      statusDot.className = 'status-dot online';
      statusText.textContent = 'Model ready';
    } else {
      statusDot.className = 'status-dot no-model';
      statusText.textContent = 'Model not trained';
    }
  } catch {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'API offline';
  }
}

// ── Load metrics ────────────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    const data = await res.json();
    if (data.available) {
      setMetric('metricAccuracy', data.accuracy  != null ? (data.accuracy  * 100).toFixed(1) + '%' : '—');
      setMetric('metricF1',       data.f1_macro   != null ? data.f1_macro.toFixed(3)           : '—');
      setMetric('metricAUC',      data.roc_auc    != null ? data.roc_auc.toFixed(3)            : '—');
    }
  } catch { /* silently skip */ }
}

function setMetric(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Load examples ────────────────────────────────────────────────────────────
async function loadExamples() {
  try {
    const res = await fetch(`${API_BASE}/examples`);
    const data = await res.json();
    exampleList.innerHTML = '';
    data.examples.forEach(ex => {
      const li = document.createElement('li');
      li.className = 'example-item';
      li.innerHTML = `
        <span class="example-text">${ex.text}</span>
        <span class="example-tag ${ex.expected === 'POSITIVE' ? 'pos' : 'neg'}">${ex.expected}</span>
      `;
      li.addEventListener('click', () => {
        textInput.value = ex.text;
        textInput.dispatchEvent(new Event('input'));
        examplesModal.hidden = true;
        setTimeout(() => btnAnalyze.click(), 120);
      });
      exampleList.appendChild(li);
    });
  } catch { /* silently skip */ }
}

// ── Main analyze function ────────────────────────────────────────────────────
async function analyze() {
  const text = textInput.value.trim();
  if (!text) return;

  setLoading(true);
  hideError();

  try {
    const res = await fetch(`${API_BASE}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
    resultsContainer.hidden = false;
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    showError(err.message || 'Something went wrong. Is the API running?');
  } finally {
    setLoading(false);
  }
}

// ── Render results ──────────────────────────────────────────────────────────
function renderResults(data) {
  renderPrediction(data.prediction, data.text);
  renderAttention(data.attention.tokens, data.attention.scores, data.prediction.label);
  renderLime(data.lime.word_scores, data.prediction.label);
}

// Prediction card
function renderPrediction(pred, text) {
  const isPos = pred.label === 'POSITIVE';
  const badge  = document.getElementById('sentimentBadge');
  const icon   = document.getElementById('sentimentIcon');
  const label  = document.getElementById('sentimentLabel');
  const pTime  = document.getElementById('processingTime');

  badge.className = `sentiment-badge ${isPos ? 'positive' : 'negative'}`;
  icon.textContent  = isPos ? '😊' : '😞';
  label.textContent = pred.label;
  pTime.textContent = `${pred.processing_time_ms}ms`;

  // Gauge
  const conf = pred.confidence;
  const totalArc = 251.2;
  const offset = totalArc - conf * totalArc;
  const arc = document.getElementById('gaugeArc');
  arc.style.strokeDashoffset = offset;
  arc.style.stroke = isPos ? '#10b981' : '#ef4444';
  document.getElementById('gaugePct').textContent = `${(conf * 100).toFixed(1)}%`;

  // Probability bars
  const probNeg = pred.probabilities['NEGATIVE'] ?? 0;
  const probPos = pred.probabilities['POSITIVE'] ?? 0;
  document.getElementById('probNeg').style.width = `${probNeg * 100}%`;
  document.getElementById('probPos').style.width = `${probPos * 100}%`;
  document.getElementById('probNegVal').textContent = `${(probNeg * 100).toFixed(1)}%`;
  document.getElementById('probPosVal').textContent = `${(probPos * 100).toFixed(1)}%`;
}

// Attention heatmap
function renderAttention(tokens, scores, label) {
  const container = document.getElementById('attentionContainer');
  container.innerHTML = '';

  if (!tokens || tokens.length === 0) {
    container.innerHTML = '<span style="color:var(--text-muted);font-size:.85rem;">No attention data available.</span>';
    return;
  }

  const maxScore = Math.max(...scores);
  tokens.forEach((token, i) => {
    const score = scores[i] ?? 0;
    const norm  = maxScore > 0 ? score / maxScore : 0;

    // Interpolate bg color: dark → warm yellow → red
    const bg    = scoreToColor(norm);
    const fg    = norm > 0.5 ? '#000' : '#f1f5f9';

    const chip = document.createElement('span');
    chip.className = 'token-chip';
    chip.style.background = bg;
    chip.style.color = fg;
    chip.setAttribute('data-score', `attn: ${score.toFixed(3)}`);
    chip.textContent = token;
    container.appendChild(chip);
  });
}

function scoreToColor(t) {
  // t in [0,1] → low=#1e293b → mid=#fbbf24 → high=#ef4444
  if (t < 0.5) {
    const u = t * 2;
    return lerpColor('#1e293b', '#fbbf24', u);
  } else {
    const u = (t - 0.5) * 2;
    return lerpColor('#fbbf24', '#ef4444', u);
  }
}

function lerpColor(a, b, t) {
  const parse = h => [
    parseInt(h.slice(1,3),16),
    parseInt(h.slice(3,5),16),
    parseInt(h.slice(5,7),16),
  ];
  const [ar,ag,ab] = parse(a);
  const [br,bg,bb] = parse(b);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bv= Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${bv})`;
}

// LIME bar chart
function renderLime(wordScores, label) {
  const container = document.getElementById('limeChart');
  container.innerHTML = '';

  if (!wordScores || wordScores.length === 0) {
    container.innerHTML = '<span style="color:var(--text-muted);font-size:.85rem;">No LIME data available.</span>';
    return;
  }

  const maxAbs = Math.max(...wordScores.map(([, s]) => Math.abs(s)));

  // Sort by absolute score desc
  const sorted = [...wordScores].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  sorted.forEach(([word, score]) => {
    const pct = maxAbs > 0 ? (Math.abs(score) / maxAbs) * 50 : 0; // 50% = half-bar each side
    const isPos = score >= 0;

    const row = document.createElement('div');
    row.className = 'lime-row';
    row.innerHTML = `
      <span class="lime-word" title="${word}">${word}</span>
      <div class="lime-track">
        <div class="lime-fill ${isPos ? 'pos' : 'neg'}" style="width:${pct}%"></div>
      </div>
      <span class="lime-score" style="color:${isPos ? 'var(--accent-green)' : 'var(--accent-red)'}">
        ${score >= 0 ? '+' : ''}${score.toFixed(3)}
      </span>
    `;
    container.appendChild(row);
  });
}

// ── UI helpers ───────────────────────────────────────────────────────────────
function setLoading(loading) {
  btnAnalyze.disabled = loading || textInput.value.trim().length < 2;
  btnAnalyze.querySelector('.btn-text').hidden = loading;
  btnAnalyze.querySelector('.btn-spinner').hidden = !loading;
}

function showError(msg) {
  toastMsg.textContent = msg;
  errorToast.hidden = false;
  setTimeout(() => { errorToast.hidden = true; }, 6000);
}

function hideError() { errorToast.hidden = true; }
