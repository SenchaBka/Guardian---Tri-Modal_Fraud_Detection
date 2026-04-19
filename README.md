# Guardian – Tri-Modal Fraud Detection System

**Multi-modal fraud detection prototype combining Numerical, NLP, and Voice streams with an Adaptive Ensemble Fusion Layer.**

This repository contains the Iteration #2 prototype for **Guardian**, a resilient fraud detection system that dynamically adapts to available data modalities (Numerical + NLP + Voice). The Fusion Layer acts as the central decision engine, enabling detection even when some modalities are missing (e.g., no voice data for online transactions).

---

## Project Overview

Guardian addresses single-modality gaps in traditional fraud detection by fusing three independent streams:

- **Numerical Stream** (Ivan) – Transaction patterns, velocity, anomalies
- **NLP Stream** (Luis) – Text analysis (merchant names, narratives, typosquatting)
- **Voice Stream** (Arsenii) – Deepfake detection, speaker consistency, stress indicators

The **Fusion Layer** (Sherwayne) orchestrates these streams using an **Adaptive Ensemble Architecture** that provides graceful degradation, dynamic weight renormalization, confidence adjustment, and SHAP-based explainability.

### Key Features of the Adaptive Ensemble
- Works with any combination of modalities (full → partial → fallback)
- Numerical stream is always required and serves as the baseline
- Weights automatically renormalize when modalities are unavailable
- Confidence penalties trigger conservative decisions (more human reviews in degraded modes)
- SHAP explanations adapt to available signals with clear notes on missing modalities

---

## Team & Ownership

| Stream / Component       | Owner      | Responsibility                          |
|--------------------------|------------|-----------------------------------------|
| Fusion / Decision Layer  | Sherwayne  | Orchestration, adaptive ensemble, SHAP  |
| NLP Stream               | Luis       | Text processing, FinBERT, API endpoint  |
| Numerical Stream         | Ivan       | Feature engineering, XGBoost/RF, API    |
| Voice Stream             | Arsenii    | Audio preprocessing, WavLM, API         |

---

## Architecture Highlights

### Modality Priority & Ensemble Modes

| Mode     | Modalities Used              | Base Confidence | Weight Example                           |
|----------|------------------------------|-----------------|------------------------------------------|
| full     | Numerical + NLP + Voice      | 100%            | Num: 0.45, NLP: 0.35, Voice: 0.20        |
| partial  | Numerical + NLP              | 85%             | Num: 0.5625, NLP: 0.4375 (renormalized)  |
| partial  | Numerical + Voice            | 80%             | Num: 0.70, Voice: 0.30 (renormalized)    |
| fallback | Numerical only               | 60%             | Num: 1.00                                |

### Weight Renormalization
When a modality is missing, weights are divided by the sum of available base weights to maintain a total of 1.0.

### Confidence Penalties

| Scenario                     | Penalty | Resulting Confidence |
|------------------------------|---------|----------------------|
| All 3 modalities             | 0%      | 100%                 |
| Voice missing                | 15%     | 85%                  |
| NLP missing                  | 20%     | 80%                  |
| Voice + NLP missing          | 40%     | 60%                  |

### Decision Thresholds

| Decision | Fraud Score Range | Default Action   | Fallback/Partial Overrides                          |
|----------|-------------------|------------------|-----------------------------------------------------|
| approve  | 0.00 – 0.30       | Auto-approve     | None                                                |
| review   | 0.31 – 0.70       | Manual review    | Forced in fallback > 0.25 or low-confidence partial |
| block    | 0.71 – 1.00       | Auto-block       | Forced in fallback > 0.60                           |

---

## Repository Structure

```
Guardian/
├── fusion/               # Fusion Layer (Sherwayne)
│   ├── api.py
│   ├── orchestrator.py
│   ├── ensemble.py
│   ├── confidence.py
│   ├── explainer.py
│   ├── audit.py
│   ├── schemas.py
│   ├── config.py
│   └── tests/
├── NPL/                  # NLP Stream (Luis)
│   ├── api/
│   ├── classifier.py
│   ├── model_loader.py
│   ├── preprocessor.py
│   └── requirements.txt
├── Voice/                # Voice Stream (Arsenii)
│   ├── api.py
│   ├── model_loader.py
│   ├── config.py
│   └── requirements.txt
├── guardian-dashboard/   # React Frontend (Sherwayne)
│   ├── src/
│   ├── vite.config.js
│   └── package.json
├── start_guardian.bat    # One-click launcher (Windows)
├── requirements.txt
└── README.md
```

---

## Port Configuration

| Service          | Owner    | Port |
|------------------|----------|------|
| Voice Stream API | Arsenii  | 8000 |
| NLP Stream API   | Luis     | 8001 |
| Numerical API    | Ivan     | 8002 |
| Fusion Layer API | Sherwayne| 8080 |
| Frontend         | Sherwayne| 3000 |

---

## Running the Full System

### Prerequisites

- Python 3.12+
- Node.js 18+
- Virtual environment with all dependencies installed
- Internet connection (first run only — models download from Hugging Face automatically)

### Step 1 — Install Dependencies

From the repository root, install all Python dependencies:

```bash
pip install -r requirements.txt
pip install -r NPL/requirements.txt
pip install -r Voice/requirements.txt
```

Install frontend dependencies:

```bash
cd guardian-dashboard
npm install
cd ..
```

### Step 2 — Run All Services (Windows — One Click)

Double-click `start_guardian.bat` in the project root. It will open 4 terminals automatically:

- **Fusion Layer** → http://localhost:8080
- **NLP Stream** → http://localhost:8001
- **Voice Stream** → http://localhost:8000
- **Frontend** → http://localhost:3000

> **Note:** The first time you run it, the NLP and Voice servers will download their models from Hugging Face. This may take a few minutes. After that, models are cached locally and startup is instant.

### Step 2 (Alternative) — Run Each Service Manually

Open four separate terminals from the project root:

**Terminal 1 — Fusion Layer:**
```bash
uvicorn fusion.api:app --reload --port 8080
```

**Terminal 2 — NLP Stream:**
```bash
python -m uvicorn NPL.api.api:app --reload --port 8001
```

**Terminal 3 — Voice Stream:**
```bash
uvicorn Voice.api:app --reload --port 8000
```

**Terminal 4 — Frontend:**
```bash
cd guardian-dashboard
npm run dev
```

### Step 3 — Open the Dashboard

Navigate to:
```
http://localhost:3000
```

---

## API Endpoints

### Fusion Layer

| Method | Endpoint               | Description                        |
|--------|------------------------|------------------------------------|
| GET    | `/api/v1/fusion/health`| Health check                       |
| POST   | `/api/v1/fusion/evaluate` | Run tri-modal fraud analysis    |
| GET    | `/api/v1/audit`        | Retrieve full audit log            |

### NLP Stream (Luis)

| Method | Endpoint               | Description                        |
|--------|------------------------|------------------------------------|
| GET    | `/health`              | Model health check                 |
| POST   | `/api/v1/nlp/score`    | Score fraud risk from text         |

**Request:**
```json
{
  "transaction_id": "txn-123",
  "language": "en",
  "payload": {
    "merchant_text": "ACME invoice payment",
    "narrative_text": "urgent refund request",
    "invoice_text": null,
    "ticket_text": null
  },
  "metadata": {
    "amount": 1299.99,
    "currency": "CAD",
    "country": "CA",
    "channel": "online",
    "transaction_type": "transfer"
  }
}
```

**Response:**
```json
{
  "transaction_id": "txn-123",
  "score_nlp": 0.91,
  "model_version": "Lmateosl/guardian-finbert-npl",
  "status": "ok",
  "signals": {
    "semantic_risk": 0.91,
    "typosquatting_risk": null,
    "entity_inconsistency": null,
    "threshold_used": 0.0039,
    "predicted_fraud": 1.0
  }
}
```

### Voice Stream (Arsenii)

| Method | Endpoint                  | Description                        |
|--------|---------------------------|------------------------------------|
| POST   | `/api/v1/voice/score`     | Deepfake detection score           |
| POST   | `/api/v1/voice/transcribe`| Speech to text via ElevenLabs      |

**Response:**
```json
{
  "label": "SPOOF",
  "bonafide_prob": 0.000004,
  "spoof_prob": 0.999996,
  "threshold": 0.162777,
  "filename": "sample_fake.flac"
}
```

---

## NLP Stream — Model Details

The NLP stream scores fraud risk from transaction text using a fine-tuned FinBERT classifier.

### Model Serving Strategy

Loading priority:
1. Hugging Face hosted model (`Lmateosl/guardian-finbert-npl`)
2. Local checkpoint in `models/nlp/finbert/paysim_sample100k_ep2`
3. Heuristic fallback if model inference fails

### Why PR-AUC

PaySim is highly imbalanced with a very small fraud class. PR-AUC was chosen over ROC-AUC because it focuses on precision and recall for the rare positive class and reduces the risk of overvaluing majority-class performance.

### Re-Generating the NLP Model (PaySim dataset required)

```bash
# Step 1 — Create sample dataset
python3 -m NPL.training.sample_paysim \
  --input-path NPL/data/interim/paysim/paysim_nlp_interim.csv \
  --output-path NPL/data/processed/paysim/paysim_sample_100k.csv \
  --sample-size 100000

# Step 2 — Fine-tune FinBERT (2 epochs recommended)
python3 -m NPL.training.train_finbert_paysim \
  --input NPL/data/processed/paysim/paysim_sample_100k.csv \
  --model-dir models/nlp/finbert/paysim_sample100k_ep2 \
  --reports-dir reports/nlp/paysim_sample100k_ep2 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --max-length 128
```

### Running NLP Tests

```bash
python3 -m unittest discover -s NPL/tests -v
```

---

## Voice Stream — Model Details

The Voice stream uses WavLM (`microsoft/wavlm-large`) with a logistic regression classifier trained on ASVspoof 2019 to detect deepfake audio.

### Model Serving Strategy

Models are hosted on Hugging Face (`senchabka/asv-spoof-detector`) and downloaded automatically on first startup.

---

## Fusion Layer — Running Tests

```bash
python -m pytest fusion/tests/ -v
```

186 tests across 5 files: `test_api.py`, `test_ensemble.py`, `test_confidence.py`, `test_explainer.py`, `test_audit.py`.

---

## Demo Scenarios

### 🟢 APPROVE — Clean Transaction
- Amount: `$45` · Country: `US` · Category: `retail`
- Merchant: `Amazon Canada` · Narrative: `online shopping`

### 🟡 REVIEW — Borderline
- Amount: `$3500` · Country: `BR` · Category: `electronics`
- Merchant: `Apple Store` · Narrative: `device purchase`

### 🔴 BLOCK — Fraud
- Amount: `$9500` · Country: `NG` · Category: `crypto`
- Merchant: `Amaz0n Electronics` · Narrative: `urgent bitcoin transfer winner`
- Voice: Deepfake sample