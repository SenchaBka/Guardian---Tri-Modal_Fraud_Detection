# NPL Internal Guide

This document is the internal reference for the `NPL` module in Guardian. It explains the data pipeline, training scripts, serving backends, test suite, and the current decisions we made while evolving the NLP stream.

## Purpose

The `NPL` module is the NLP stream of the Guardian tri-modal fraud detection system. Its job is to:

- receive transaction text from the Fusion layer
- preprocess and normalize text into a canonical internal format
- produce an NLP fraud risk score
- expose the score through a FastAPI service
- support offline experimentation with both classical and transformer-based models

Today, the module supports two serving backends:

- `transformer`: FinBERT-based fraud classifier
- `sklearn`: TF-IDF + classical baseline such as `LinearSVM`

## Folder Map

Key files and folders:

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/api/api.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/api/api.py)
  FastAPI entrypoint for the NLP service.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/api/schemas.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/api/schemas.py)
  Runtime API contracts for requests and responses.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/preprocessor.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/preprocessor.py)
  Converts `RawNLPRequest` into canonical `NLPInput`.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/model_loader.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/model_loader.py)
  Loads the active NLP backend from Hugging Face or a local fallback directory.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/classifier.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/classifier.py)
  Inference logic and heuristic fallback.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/notebooks/01_paysim_eda_nlp.ipynb`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/notebooks/01_paysim_eda_nlp.ipynb)
  PaySim EDA and interim dataset creation.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/notebooks/01_ieee_eda_nlp.ipynb`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/notebooks/01_ieee_eda_nlp.ipynb)
  IEEE EDA and interim dataset creation.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training)
  Offline sampling, merging, baseline training, FinBERT training, and evaluation helpers.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests)
  Unit tests for API, loader, classifier, preprocessing, and training utilities.

## Runtime Contracts

The external API uses `RawNLPRequest` and returns `NLPOutput`.

Runtime flow:

1. Fusion sends `RawNLPRequest`
2. `preprocessor.py` creates canonical `NLPInput`
3. `classifier.py` scores fraud risk
4. `api.py` returns `NLPOutput`

Important request fields:

- `transaction_id`
- `language`
- `payload.merchant_text`
- `payload.narrative_text`
- optional metadata such as `amount`, `country`, `channel`, and `transaction_type`

Main response fields:

- `transaction_id`
- `score_nlp`
- `model_version`
- `status`
- `signals.semantic_risk`
- `signals.threshold_used`
- `signals.predicted_fraud`

## Data Pipeline

We currently use two sources:

- PaySim
- IEEE-CIS

Both notebooks were aligned to export the same interim schema:

- `transaction_id`
- `text`
- `label`
- `amount`
- `transaction_type`
- `dataset_source`

Where:

- `label = 1` means fraud
- `label = 0` means non-fraud
- `dataset_source` is `paysim` or `ieee`

This alignment was necessary so both datasets could be merged into one training pipeline.

### Interim and processed datasets

Typical files:

- `NPL/data/interim/paysim/paysim_nlp_interim.csv`
- `NPL/data/interim/ieee/ieee_nlp_interim.csv`
- `NPL/data/processed/paysim/paysim_sample_100k.csv`
- `NPL/data/processed/ieee/ieee_sample_100k.csv`
- `NPL/data/processed/combined/nlp_combined_200k.csv`
- `NPL/data/processed/combined/nlp_combined_100k.csv`

### Scripts used in the data pipeline

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/sample_paysim.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/sample_paysim.py)
  Creates a stratified PaySim sample.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/sample_ieee.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/sample_ieee.py)
  Creates a stratified IEEE sample.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/sample_dataset.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/sample_dataset.py)
  Creates a stratified sample from any aligned dataset, including the combined CSV.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/merge_interim_datasets.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/merge_interim_datasets.py)
  Merges full interim CSVs.

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/merge_sampled_datasets.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/merge_sampled_datasets.py)
  Merges sampled PaySim and IEEE datasets.

## Training Backends

We used two main training families:

### 1. Classical baseline

Implemented in:

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/train_baseline.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/train_baseline.py)

Supported models:

- `logreg`
- `linear_svm`

This pipeline uses:

- `TfidfVectorizer`
- validation threshold search
- `PR-AUC`, `Recall`, `F1`, `ROC-AUC`

Example:

```bash
python3 -m NPL.training.train_baseline \
  --input NPL/data/processed/combined/nlp_combined_200k.csv \
  --dataset-name paysim_ieee_combined \
  --run-name combined_200k_linear_svm \
  --model-type linear_svm
```

Extra tuning options already supported:

- `--svm-c`
- `--sublinear-tf`
- `--min-df`
- `--max-df`
- `--ngram-max`

### 2. FinBERT fraud classifier

Implemented in:

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/train_finbert.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/training/train_finbert.py)

This pipeline uses:

- `ProsusAI/finbert`
- class-weighted loss
- threshold search on validation
- optional small grid search

Example final run:

```bash
python3 -m NPL.training.train_finbert \
  --input NPL/data/processed/combined/nlp_combined_100k.csv \
  --dataset-name paysim_ieee_combined \
  --run-name combined_100k_best_ep2 \
  --num-train-epochs 2 \
  --learning-rate 3e-5 \
  --max-length 128
```

### Why we prioritized PR-AUC

The fraud datasets are highly imbalanced. Because of that, `PR-AUC` was treated as the main model-selection metric. This makes the evaluation more honest for fraud detection than relying only on `accuracy` or even `ROC-AUC`.

Still, we also compare:

- `Recall`
- `F1`
- final operating threshold

This matters because one model can win on `PR-AUC` while another performs better at a specific thresholded decision point.

## Current Experimental Findings

At the time of writing:

- `LogisticRegression` remains the strongest classical baseline on the combined dataset in terms of `Recall` and `F1`
- `LinearSVM` did not beat `LogisticRegression`
- `FinBERT` achieved the strongest `PR-AUC` on the combined dataset among the tested models

Interpretation:

- `FinBERT` is stronger for ranking fraud risk under the chosen metric
- `LogisticRegression` is still competitive, especially as a simple classical baseline
- `LinearSVM` works better when inputs resemble the training-domain text style, but is not currently the best overall baseline

## Serving the API

The NLP API lives in:

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/api/api.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/api/api.py)

Run:

```bash
python3 -m uvicorn NPL.api.api:app --reload --port 8001
```

Swagger:

- [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

Health:

- [http://127.0.0.1:8001/health](http://127.0.0.1:8001/health)

### Serving FinBERT

Example:

```bash
export NLP_MODEL_KIND=transformer
export NLP_MODEL_NAME=Lmateosl/guardian-finbert-npl
export NLP_MODEL_REVISION=main
python3 -m uvicorn NPL.api.api:app --reload --port 8001
```

Or use a local fallback:

```bash
export NLP_MODEL_KIND=transformer
export NLP_MODEL_DIR=models/nlp/finbert/combined_100k_best_ep2
python3 -m uvicorn NPL.api.api:app --reload --port 8001
```

### Serving LinearSVM

Example with local artifacts:

```bash
export NLP_MODEL_KIND=sklearn
export NLP_MODEL_DIR=models/nlp/baseline/combined_200k_linear_svm
python3 -m uvicorn NPL.api.api:app --reload --port 8001
```

Example with a Hugging Face model repo:

```bash
export NLP_MODEL_KIND=sklearn
export NLP_MODEL_NAME=Lmateosl/guardian-linear-svm-npl
export NLP_MODEL_REVISION=main
python3 -m uvicorn NPL.api.api:app --reload --port 8001
```

### Health endpoint fields

`GET /health` now reports:

- `status`
- `model`
- `threshold`
- `model_version`
- `device`
- `model_source`
- `model_revision`
- `model_backend`

This is the easiest way to confirm whether the API is serving:

- Hugging Face or local artifacts
- FinBERT or sklearn backend

## Hugging Face Model Uploads

For large or shareable artifacts, upload the served bundle to Hugging Face instead of committing it to Git.

Example for the current `LinearSVM` bundle:

```bash
hf auth login
hf upload Lmateosl/guardian-linear-svm-npl models/nlp/baseline/combined_200k_linear_svm . --type model
```

Expected files in the repo:

- `linear_svm_model.joblib`
- `tfidf_vectorizer.joblib`
- `metadata.json`

## Tests

Test suite:

- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests/test_api.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests/test_api.py)
- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests/test_classifier_and_loader.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests/test_classifier_and_loader.py)
- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests/test_preprocessor.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests/test_preprocessor.py)
- [`/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests/test_training_utils.py`](/Users/mateoff/Desktop/Centennial/6-semestre/capstone/project/Guardian---Tri-Modal_Fraud_Detection/NPL/tests/test_training_utils.py)

Run:

```bash
python3 -m unittest discover -s NPL/tests -v
```

## Known Limitations

- `LinearSVM` behaves more reliably on in-domain, structured text than on free-form text written by hand
- `FinBERT` is heavier to serve but stronger semantically
- thresholded classification quality and ranking quality do not always point to the same best model
- PaySim-only results are much easier than combined PaySim+IEEE results, so they should not be interpreted the same way

## Recommended Default

For internal experimentation:

- keep `FinBERT` as the main semantic model candidate
- keep `LogisticRegression` as the reference classical baseline
- keep `LinearSVM` as an additional classical experiment or controlled serving backend when inputs are close to the training-domain text format
