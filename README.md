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

