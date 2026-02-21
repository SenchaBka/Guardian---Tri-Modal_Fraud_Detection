"""
trainer.py
----------
End-to-end training pipeline for the Guardian Numerical Stream.

Pipeline steps (numerical_stream_contract.docx §4):
  1. Load labelled transaction data
  2. Feature engineering
  3. Train/validation split (stratified)
  4. SMOTE oversampling on training set only
  5. Train XGBoost with scale_pos_weight + Platt Scaling calibration
  6. Train Isolation Forest for pattern_deviation signal
  7. Tune decision threshold on validation set (max F1)
  8. Evaluate on held-out test set
  9. Persist model artefacts to disk
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from feature_engineering import extract_features
from models import (
    MODEL_VERSION,
    evaluate_model,
    train_isolation_forest,
    train_random_forest,
    train_xgboost,
    tune_decision_threshold,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "numerical_datasets"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

XGB_MODEL_PATH = MODELS_DIR / "xgboost_fraud.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf_fraud.joblib"
ISO_FOREST_PATH = MODELS_DIR / "isolation_forest.joblib"
THRESHOLD_PATH = MODELS_DIR / "decision_threshold.joblib"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset(
    data_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the training dataset and return (X, y).

    Tries to load from `data_path` (CSV expected with a 'label' column).
    Falls back to a synthetic dataset for development / CI purposes.

    Parameters
    ----------
    data_path : path to a CSV file with transaction features + 'label' column

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)  — binary (0=legit, 1=fraud)
    """
    if data_path is not None and Path(data_path).exists():
        import pandas as pd
        logger.info("Loading dataset from %s", data_path)
        df = pd.read_csv(data_path)
        y = df["label"].values.astype(int)
        X = df.drop(columns=["label"]).values.astype(np.float32)
        logger.info("Dataset loaded: %d samples, %d features, %.4f fraud rate",
                    len(y), X.shape[1], y.mean())
        return X, y

    logger.warning("No dataset found — generating synthetic data for development.")
    return _generate_synthetic_dataset()


def _generate_synthetic_dataset(
    n_legit: int = 50_000,
    n_fraud: int = 250,   # ~0.5 % fraud rate
    n_features: int = 16,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a small synthetic dataset to allow the pipeline to run without real data."""
    rng = np.random.RandomState(random_state)
    X_legit = rng.randn(n_legit, n_features).astype(np.float32)
    X_fraud = rng.randn(n_fraud, n_features).astype(np.float32) + 2.0  # shifted distribution
    X = np.vstack([X_legit, X_fraud])
    y = np.hstack([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# SMOTE helper
# ---------------------------------------------------------------------------

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to the *training set only* to balance the fraud class.
    NEVER apply to evaluation / test data (numerical_stream_contract.docx §4.2).
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imbalanced-learn not installed — skipping SMOTE. Install with: pip install imbalanced-learn")
        return X, y

    fraud_count = int(y.sum())
    legit_count = int(len(y) - fraud_count)
    logger.info("Before SMOTE: %d legit, %d fraud", legit_count, fraud_count)

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    logger.info("After SMOTE:  %d legit, %d fraud", int((y_res == 0).sum()), int((y_res == 1).sum()))
    return X_res, y_res


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    data_path: Optional[Path] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    use_smote: bool = True,
    save_models: bool = True,
) -> Dict:
    """
    Full end-to-end training pipeline.

    Steps:
      1. Load data
      2. Stratified train / val / test split
      3. Optionally apply SMOTE to training set
      4. Train XGBoost (calibrated) + Random Forest + Isolation Forest
      5. Tune decision threshold on validation set
      6. Evaluate on test set
      7. Save artefacts

    Parameters
    ----------
    data_path   : path to CSV dataset (None → synthetic data)
    test_size   : proportion of data held out for final evaluation
    val_size    : proportion used for threshold tuning (of remaining data)
    use_smote   : whether to apply SMOTE to training set
    save_models : whether to persist models to disk

    Returns
    -------
    dict with test metrics and saved model paths
    """
    logger.info("=== Guardian Numerical Stream — Training Pipeline ===")

    # 1. Load data
    X, y = load_dataset(data_path)
    scale_pos_weight = int((y == 0).sum() / max((y == 1).sum(), 1))
    logger.info("scale_pos_weight = %d", scale_pos_weight)

    # 2. Stratified splits:  train+val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    # train / val split
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=42
    )
    logger.info("Split sizes — train: %d  val: %d  test: %d", len(y_train), len(y_val), len(y_test))

    # 3. SMOTE (training set only)
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # 4a. Train XGBoost (calibrated with Platt Scaling)
    xgb_model = train_xgboost(X_train, y_train, scale_pos_weight=scale_pos_weight, calibrate=True)

    # 4b. Train Random Forest
    rf_model = train_random_forest(X_train, y_train)

    # 4c. Train Isolation Forest on all training data (unsupervised)
    iso_model = train_isolation_forest(X_train, contamination=0.005)

    # 5. Tune decision threshold on validation set
    optimal_threshold = tune_decision_threshold(xgb_model, X_val, y_val)

    # 6. Evaluate on test set
    logger.info("=== Test Set Evaluation ===")
    test_metrics = evaluate_model(xgb_model, X_test, y_test)
    test_metrics["optimal_threshold"] = optimal_threshold

    # 7. Save artefacts
    saved_paths = {}
    if save_models:
        joblib.dump(xgb_model, XGB_MODEL_PATH)
        joblib.dump(rf_model, RF_MODEL_PATH)
        joblib.dump(iso_model, ISO_FOREST_PATH)
        joblib.dump(optimal_threshold, THRESHOLD_PATH)
        saved_paths = {
            "xgboost": str(XGB_MODEL_PATH),
            "random_forest": str(RF_MODEL_PATH),
            "isolation_forest": str(ISO_FOREST_PATH),
            "threshold": str(THRESHOLD_PATH),
        }
        logger.info("Models saved to %s", MODELS_DIR)

    logger.info("=== Training Complete ===")
    logger.info("Test metrics: %s", test_metrics)

    return {"metrics": test_metrics, "model_version": MODEL_VERSION, "paths": saved_paths}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Guardian Numerical Stream models")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV training data")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--no-save", action="store_true", help="Do not save model artefacts")
    args = parser.parse_args()

    results = run_training_pipeline(
        data_path=Path(args.data) if args.data else None,
        use_smote=not args.no_smote,
        save_models=not args.no_save,
    )
    print("\nFinal results:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}")
