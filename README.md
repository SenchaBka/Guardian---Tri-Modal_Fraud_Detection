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

The NLP stream focuses on detecting fraud patterns from textual transaction narratives using a fine-tuned transformer model.

### Model & Approach

- Base model: **ProsusAI/finbert**
- Task: Binary classification (fraud vs non-fraud)
- Dataset: **PaySim (synthetic financial transactions)**
- Input: Structured narrative text generated from transaction fields (type, amount, balances)
- Output: Fraud probability (`score_nlp`) used by the Fusion Layer

### Key Findings

- The model performs extremely well on **PaySim-style structured narratives**
- It learns strong signals from:
  - Transaction type (e.g., CASH_OUT)
  - Balance inconsistencies
  - Large transfers and zeroing balances
- However, it does **not generalize well to free-form human text** (e.g., complaints or emails)
- This highlights the need for **multi-dataset training (e.g., IEEE, FUNSD)** in future iterations

### Models Trained

Three main configurations were explored:

#### 1. Baseline (Pretrained FinBERT)
- No fine-tuning
- Used as reference
- Poor performance on PaySim task

#### 2. FinBERT – 2 Epochs (Best Practical Model)
- Fast training
- Good generalization
- Selected for API integration

#### 3. FinBERT – 3 Epochs
- Slightly better training metrics
- Risk of overfitting due to extreme class imbalance (fraud rate ~0.13%)
- Used for comparison

### Training Commands

All models are trained using the same pipeline:

#### 2 Epoch Model (Recommended)
```
python3 -m NPL.training.train_finbert_paysim \
  --input NPL/data/processed/paysim/paysim_sample_100k.csv \
  --model-dir models/nlp/finbert/paysim_sample100k_ep2 \
  --reports-dir reports/nlp/paysim_sample100k_ep2 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --max-length 128
```

#### 3 Epoch Model
```
python3 -m NPL.training.train_finbert_paysim \
  --input NPL/data/processed/paysim/paysim_sample_100k.csv \
  --model-dir models/nlp/finbert/paysim_sample100k_ep3 \
  --reports-dir reports/nlp/paysim_sample100k_ep3 \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --max-length 128
```

#### Baseline (No Fine-Tuning)
- Directly uses `ProsusAI/finbert`
- No training command required

### Outputs

Each training run generates:

- Model checkpoints → `models/nlp/...`
- Evaluation reports → `reports/nlp/...`
- Metrics include:
  - Precision / Recall / F1
  - ROC-AUC / PR-AUC
  - Confusion Matrix
  - Best threshold (used later in API)

### API Integration

The trained model is exposed through a FastAPI endpoint:

- Input: standardized NLP contract (transaction_id, text, metadata)
- Output:
  - `score_nlp` (fraud probability)
  - `predicted_fraud` (binary using learned threshold)
  - additional signals for fusion layer

The API uses the **best validation threshold automatically extracted during training**.

### Running Tests (NLP)

To execute the unit tests for the NLP stream:

```
python3 -m unittest discover -s NPL/tests -v
```

This will run all NLP-related test cases including classifier, API, preprocessing, and training utilities.

---
