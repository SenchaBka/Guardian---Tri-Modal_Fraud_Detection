"""
models.py
---------
Model definitions and training logic for the Guardian Numerical Stream.

Architecture (numerical_stream_contract.docx §4):
  - Base classifier : XGBoost (XGBClassifier)
  - Calibration     : Platt Scaling (CalibratedClassifierCV, method='sigmoid')
  - Class imbalance : scale_pos_weight + SMOTE + stratified K-Fold CV
  - Threshold tuning: optimise on validation F1 for fraud class

Performance targets (numerical_stream_contract.docx §7):
  Precision  >= 0.85
  Recall     >= 0.80
  F1         >= 0.82
  AUC-ROC    >= 0.95
  FPR        <= 0.02
  Latency P95 <= 50 ms
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "xgb-v1.0.0"

# XGBoost hyper-parameters (tunable)
XGB_PARAMS: Dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

RANDOM_FOREST_PARAMS: Dict = {
    "n_estimators": 200,
    "max_depth": 8,
    "n_jobs": -1,
    "random_state": 42,
}

# Fraud class is rare (~0.5 % of transactions) → scale_pos_weight ≈ 200
DEFAULT_SCALE_POS_WEIGHT = 200


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    scale_pos_weight: int = DEFAULT_SCALE_POS_WEIGHT,
    calibrate: bool = True,
    n_cv_folds: int = 5,
) -> CalibratedClassifierCV:
    """
    Train an XGBoost fraud classifier with Platt Scaling calibration.

    Class-imbalance handling (numerical_stream_contract.docx §4.2):
      - scale_pos_weight set to ratio of negatives to positives (~200:1)
      - Stratified K-Fold CV preserves fraud class ratio across folds
      - Threshold is tuned separately (see tune_decision_threshold)

    Parameters
    ----------
    X                 : feature matrix, shape (n_samples, n_features)
    y                 : binary labels (0 = legit, 1 = fraud)
    scale_pos_weight  : negative/positive sample ratio
    calibrate         : wrap in CalibratedClassifierCV (Platt Scaling)
    n_cv_folds        : folds for stratified CV

    Returns
    -------
    Calibrated XGBClassifier whose predict_proba outputs a well-calibrated
    fraud probability score_numerical ∈ [0, 1].
    """
    logger.info("Training XGBoost  (scale_pos_weight=%d, calibrate=%s)", scale_pos_weight, calibrate)

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    base_model = XGBClassifier(**params)

    if calibrate:
        cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
        model = CalibratedClassifierCV(base_model, method="sigmoid", cv=cv)
    else:
        model = base_model

    model.fit(X, y)
    logger.info("XGBoost training complete.")
    return model


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier as a secondary ensemble member.

    Parameters
    ----------
    X            : feature matrix
    y            : binary labels
    class_weight : 'balanced' applies inverse-frequency weighting

    Returns
    -------
    Fitted RandomForestClassifier
    """
    logger.info("Training Random Forest (class_weight=%s)", class_weight)
    params = {**RANDOM_FOREST_PARAMS, "class_weight": class_weight}
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    logger.info("Random Forest training complete.")
    return model


# ---------------------------------------------------------------------------
# Isolation Forest  (used to compute pattern_deviation signal)
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.005,
) -> IsolationForest:
    """
    Train an Isolation Forest for unsupervised anomaly scoring.

    The anomaly score is surfaced as the `pattern_deviation` sub-signal
    in the Numerical Stream API output.

    Parameters
    ----------
    X             : feature matrix (typically the full unlabelled transaction dataset)
    contamination : expected fraction of fraud/outliers (~0.5 %)

    Returns
    -------
    Fitted IsolationForest
    """
    logger.info("Training Isolation Forest (contamination=%.4f)", contamination)
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)
    logger.info("Isolation Forest training complete.")
    return model


def isolation_forest_score(model: IsolationForest, x: np.ndarray) -> float:
    """
    Convert IsolationForest decision_function output to a [0, 1] anomaly score.

    decision_function returns negative values for anomalies; we invert and
    normalise so that 1.0 = most anomalous.
    """
    raw = model.decision_function(x.reshape(1, -1))[0]
    # Raw is in roughly [-0.5, 0.5]; flip so higher = more anomalous
    score = float(np.clip(0.5 - raw, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_decision_threshold(
    model: CalibratedClassifierCV,
    X_val: np.ndarray,
    y_val: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> float:
    """
    Search for the decision threshold that maximises F1 on the fraud class
    over the validation set (numerical_stream_contract.docx §4.2).

    Parameters
    ----------
    model      : calibrated model with predict_proba
    X_val      : validation feature matrix
    y_val      : validation labels
    thresholds : array of candidate thresholds (default: 0.01 – 0.99 step 0.01)

    Returns
    -------
    Optimal threshold float in (0, 1).
    """
    from sklearn.metrics import f1_score

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    proba = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    logger.info("Best threshold=%.3f  (F1=%.4f)", best_threshold, best_f1)
    return best_threshold


# ---------------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Stratified K-Fold cross-validation for fraud detection metrics.

    Targets (numerical_stream_contract.docx §7):
        Precision  >= 0.85
        Recall     >= 0.80
        F1         >= 0.82
        AUC-ROC    >= 0.95
        FPR        <= 0.02

    Parameters
    ----------
    model    : sklearn-compatible estimator
    X        : feature matrix
    y        : labels
    n_splits : number of CV folds

    Returns
    -------
    dict of mean metric values across folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    metrics = {k: float(np.mean(v)) for k, v in results.items() if k.startswith("test_")}
    # Rename keys for clarity
    metrics = {k.replace("test_", ""): v for k, v in metrics.items()}

    _log_metric_targets(metrics)
    return metrics


def _log_metric_targets(metrics: Dict[str, float]) -> None:
    targets = {
        "precision": 0.85,
        "recall": 0.80,
        "f1": 0.82,
        "roc_auc": 0.95,
    }
    for metric, target in targets.items():
        value = metrics.get(metric, 0.0)
        status = "✓ PASS" if value >= target else "✗ FAIL"
        logger.info("  %s  %s=%.4f (target >= %.2f)", status, metric, value, target)
