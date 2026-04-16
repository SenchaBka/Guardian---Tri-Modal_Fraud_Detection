# Guardian – Tri-Modal Fraud Detection System

**Multi-modal fraud detection prototype combining Numerical, NLP, and Voice streams with an Adaptive Ensemble Fusion Layer.**

This repository contains the Iteration #1 prototype for **Guardian**, a resilient fraud detection system that dynamically adapts to available data modalities (Numerical + NLP + Voice). The Fusion Layer acts as the central decision engine, enabling detection even when some modalities are missing (e.g., no voice data for online transactions).

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

## Team & Ownership

| Stream / Component       | Owner      | Responsibility                          |
|--------------------------|------------|-----------------------------------------|
| Fusion / Decision Layer  | Sherwayne  | Orchestration, adaptive ensemble, SHAP  |
| NLP Stream               | Luis       | Text processing, FinBERT, API endpoint  |
| Numerical Stream         | Ivan       | Feature engineering, XGBoost/RF, API    |
| Voice Stream             | Arsenii    | Audio preprocessing, Wav2Vec 2.0, API   |

## Architecture Highlights

### Modality Priority & Ensemble Modes

| Mode     | Modalities Used              | Base Confidence | Weight Example                          |
|----------|------------------------------|-----------------|-----------------------------------------|
| full     | Numerical + NLP + Voice      | 100%            | Num: 0.45, NLP: 0.35, Voice: 0.20       |
| partial  | Numerical + NLP              | 85%             | Num: 0.5625, NLP: 0.4375 (renormalized) |
| partial  | Numerical + Voice            | 80%             | Num: 0.70, Voice: 0.30 (renormalized)   |
| fallback | Numerical only               | 60%             | Num: 1.00                               |

### Weight Renormalization
When a modality is missing, weights are divided by the sum of available base weights to maintain a total of 1.0.

### Confidence Penalties
| Scenario                     | Penalty | Resulting Confidence |
|------------------------------|---------|----------------------|
| All 3 modalities             | 0%      | 100%                 |
| Voice missing                | 15%     | 85%                  |
| NLP missing                  | 20%     | 80%                  |
| Voice + NLP missing          | 40%     | 60%                  |

### Decision Thresholds (with Conservative Overrides in Degraded Modes)

| Decision | Fraud Score Range | Default Action          | Fallback/Partial Overrides                     |
|----------|-------------------|-------------------------|------------------------------------------------|
| approve  | 0.00 – 0.30       | Auto-approve            | None                                           |
| review   | 0.31 – 0.70       | Manual review           | Forced in fallback > 0.25 or low-confidence partial |
| block    | 0.71 – 1.00       | Auto-block              | Forced in fallback > 0.60                      |


## Repository Structure

---

## NLP Stream (Luis)

The NLP stream scores fraud risk from transaction text using a fine-tuned FinBERT classifier wrapped in a FastAPI service.

### What The NLP Stream Does

- Validates and normalizes raw text payloads from the Fusion layer
- Combines available text fields into one canonical input
- Scores fraud probability with a hosted Hugging Face model
- Falls back to a local checkpoint if the hosted model cannot be loaded
- Falls back again to a lightweight heuristic if model loading or inference fails

### Model Serving Strategy

The API no longer depends on the `models/` folder as the primary source.

Current loading priority:

1. Hugging Face hosted model
2. Local checkpoint in `models/nlp/finbert/paysim_sample100k_ep2`
3. Heuristic fallback in code if model inference fails

Default Hugging Face configuration in [`NPL/model_loader.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/model_loader.py):

- `NLP_MODEL_NAME=Lmateosl/guardian-finbert-npl`
- `NLP_MODEL_REVISION=main`
- `NLP_THRESHOLD=0.0039`

Optional environment variables:

- `NLP_MODEL_NAME`: Hugging Face model repo id
- `NLP_MODEL_REVISION`: branch, tag, or commit
- `NLP_MODEL_DIR`: local fallback path
- `NLP_THRESHOLD`: fraud decision threshold
- `NLP_DEVICE`: `cpu`, `cuda`, or `mps`
- `NLP_CACHE_DIR`: custom Hugging Face cache directory

### If You Want To Re-Generate The Model

Training is still fully supported. The promoted Hugging Face model was produced from the PaySim pipeline in this repository.

#### 1. Create a reproducible sample dataset

Run [`NPL/training/sample_paysim.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/sample_paysim.py):

```bash
python3 -m NPL.training.sample_paysim \
  --input-path NPL/data/interim/paysim/paysim_nlp_interim.csv \
  --output-path NPL/data/processed/paysim/paysim_sample_100k.csv \
  --sample-size 100000
```

#### 2. Fine-tune FinBERT

Run [`NPL/training/train_finbert_paysim.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/train_finbert_paysim.py):

Recommended 2-epoch run:

```bash
python3 -m NPL.training.train_finbert_paysim \
  --input NPL/data/processed/paysim/paysim_sample_100k.csv \
  --model-dir models/nlp/finbert/paysim_sample100k_ep2 \
  --reports-dir reports/nlp/paysim_sample100k_ep2 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --max-length 128
```

Optional 3-epoch comparison run:

```bash
python3 -m NPL.training.train_finbert_paysim \
  --input NPL/data/processed/paysim/paysim_sample_100k.csv \
  --model-dir models/nlp/finbert/paysim_sample100k_ep3 \
  --reports-dir reports/nlp/paysim_sample100k_ep3 \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --max-length 128
```

Generated artifacts:

- Checkpoints in `models/nlp/...`
- Evaluation reports in `reports/nlp/...`
- Validation/test metrics including precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and best threshold

### Before Running The NLP API

Before starting `uvicorn`, prepare the environment so model loading and FastAPI startup do not fail.

#### 1. Create and activate a Python environment

Example with Conda:

```bash
conda create -n guardian-npl python=3.12 -y
conda activate guardian-npl
```

Or with `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install the NLP dependencies

From the repository root:

```bash
pip install -r NPL/requirements.txt
```

This installs the packages needed by the NLP API, preprocessing, tests, Hugging Face model loading, and the FinBERT training pipeline.

#### 3. Make sure the model can be resolved

The default behavior is to load the hosted Hugging Face model:

- `NLP_MODEL_NAME=Lmateosl/guardian-finbert-npl`
- `NLP_MODEL_REVISION=main`

Optional local fallback:

- Place a checkpoint in `models/nlp/finbert/paysim_sample100k_ep2`, or
- point `NLP_MODEL_DIR` to another local checkpoint directory

Optional example overrides:

```bash
export NLP_MODEL_NAME=Lmateosl/guardian-finbert-npl
export NLP_MODEL_REVISION=main
export NLP_THRESHOLD=0.0039
```

If the hosted model cannot be loaded and no valid local checkpoint exists, the API still starts, but scoring degrades to the heuristic fallback.

#### Quick Start

With `conda`:

```bash
conda create -n guardian-npl python=3.12 -y
conda activate guardian-npl
pip install -r NPL/requirements.txt
python3 -m uvicorn NPL.api.api:app --reload
```

With `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r NPL/requirements.txt
python3 -m uvicorn NPL.api.api:app --reload
```

### Running The NLP API

From the repository root:

```bash
python3 -m uvicorn NPL.api.api:app --reload
```

Default local URL:

- API base: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### API Endpoints

#### `GET /health`

Returns whether the NLP model loaded successfully and, when available, the active source, revision, device, and threshold.

#### `POST /api/v1/nlp/score`

Scores fraud risk for a transaction text payload.

Request body shape:

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

Response body shape:

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

Response field summary:

- `score_nlp`: fraud probability in `[0, 1]`
- `model_version`: active model source or local checkpoint path
- `status`: `ok` or `degraded`
- `signals.semantic_risk`: same score exposed for fusion
- `signals.threshold_used`: active decision threshold if a model was used
- `signals.predicted_fraud`: binary flag derived from the threshold when available

### Running Tests (NLP)

```bash
python3 -m unittest discover -s NPL/tests -v
```

This suite covers API behavior, preprocessing, classifier/model loading logic, and training utilities.

---
