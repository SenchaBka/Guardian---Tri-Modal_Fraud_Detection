"""
inference.py
------------
Prediction logic for the Guardian Numerical Stream.

Loads the persisted XGBoost (calibrated) and Isolation Forest models,
runs feature engineering, and returns score_numerical plus all four
sub-signals as defined in numerical_stream_contract.docx §5.

Latency targets (numerical_stream_contract.docx §9):
  P50 < 30 ms  |  P95 < 50 ms  |  P99 < 100 ms
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from feature_engineering import compute_signal_scores, extract_features
from models import MODEL_VERSION, isolation_forest_score
from schemas import NumericalScoreResponse, SignalScores, StreamStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry (module-level singletons — loaded once at startup)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_xgb_model = None
_iso_model = None
_decision_threshold: float = 0.5
_model_version: str = MODEL_VERSION
_models_loaded: bool = False


def load_models(
    xgb_path: Optional[Path] = None,
    iso_path: Optional[Path] = None,
    threshold_path: Optional[Path] = None,
) -> None:
    """
    Load persisted model artefacts into module-level singletons.
    Called once at application startup (e.g. from FastAPI lifespan).

    Falls back to placeholder (untrained) behaviour if files are not found,
    so the API remains callable during development (Iteration #1).
    """
    global _xgb_model, _iso_model, _decision_threshold, _model_version, _models_loaded

    xgb_path = xgb_path or MODELS_DIR / "xgboost_fraud.joblib"
    iso_path = iso_path or MODELS_DIR / "isolation_forest.joblib"
    threshold_path = threshold_path or MODELS_DIR / "decision_threshold.joblib"

    if Path(xgb_path).exists():
        _xgb_model = joblib.load(xgb_path)
        logger.info("XGBoost model loaded from %s", xgb_path)
    else:
        logger.warning("XGBoost model not found at %s — using placeholder.", xgb_path)

    if Path(iso_path).exists():
        _iso_model = joblib.load(iso_path)
        logger.info("Isolation Forest loaded from %s", iso_path)
    else:
        logger.warning("Isolation Forest not found at %s — using placeholder.", iso_path)

    if Path(threshold_path).exists():
        _decision_threshold = float(joblib.load(threshold_path))
        logger.info("Decision threshold loaded: %.3f", _decision_threshold)

    _models_loaded = bool(_xgb_model)


def is_model_loaded() -> bool:
    return _models_loaded


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict(features: np.ndarray) -> dict:
    """
    Run the XGBoost classifier on a pre-built feature vector and return
    the combined fraud score plus sub-signals.

    Parameters
    ----------
    features : np.ndarray of shape (n_features,)
               Produced by feature_engineering.extract_features()

    Returns
    -------
    dict:
        score_numerical  (float [0,1]) — calibrated fraud probability
        amount_anomaly   (float [0,1])
        velocity_risk    (float [0,1])
        pattern_deviation(float [0,1]) — from Isolation Forest
        geo_risk         (float [0,1])
        status           (str) — 'ok' | 'degraded' | 'error'
        model_version    (str)
    """
    if features is None or len(features) == 0:
        raise ValueError("Feature vector is empty.")

    x = features.reshape(1, -1)

    # XGBoost fraud probability
    if _xgb_model is not None:
        score_numerical = float(_xgb_model.predict_proba(x)[0][1])
    else:
        # Placeholder: use a heuristic from feature index 2 (amount_anomaly)
        score_numerical = float(np.clip(features[2] if len(features) > 2 else 0.3, 0.0, 1.0))

    # Isolation Forest anomaly score → pattern_deviation signal
    if _iso_model is not None:
        pattern_deviation = isolation_forest_score(_iso_model, features)
    else:
        pattern_deviation = float(np.clip(features[2] if len(features) > 2 else 0.3, 0.0, 1.0))

    # Extract sub-signals directly from the feature vector
    # Indices match extract_features() layout:
    #   [2] = amount_anomaly, [6] = velocity_risk, [10] = geo_risk
    amount_anomaly = float(np.clip(features[2] if len(features) > 2 else 0.0, 0.0, 1.0))
    velocity_risk = float(np.clip(features[6] if len(features) > 6 else 0.0, 0.0, 1.0))
    geo_risk = float(np.clip(features[10] if len(features) > 10 else 0.0, 0.0, 1.0))

    return {
        "score_numerical": score_numerical,
        "amount_anomaly": amount_anomaly,
        "velocity_risk": velocity_risk,
        "pattern_deviation": pattern_deviation,
        "geo_risk": geo_risk,
        "status": StreamStatus.OK,
        "model_version": _model_version,
    }


# ---------------------------------------------------------------------------
# High-level score function (used by api.py)
# ---------------------------------------------------------------------------

def score_transaction(
    transaction_id: str,
    transaction: Dict[str, Any],
    history: Optional[list] = None,
    feature_vector: Optional[np.ndarray] = None,
) -> NumericalScoreResponse:
    """
    End-to-end scoring: raw transaction dict → NumericalScoreResponse.

    Parameters
    ----------
    transaction_id : unique identifier echoed in the response
    transaction    : raw transaction dict (amount, timestamp, channel, etc.)
    history        : list of past transaction dicts for velocity/z-score
    feature_vector : optional pre-computed feature array (bypasses engineering)

    Returns
    -------
    NumericalScoreResponse — ready to be serialised by FastAPI
    """
    start_time = time.monotonic()
    status = StreamStatus.OK

    try:
        # Feature engineering
        if feature_vector is not None and len(feature_vector) > 0:
            features = np.array(feature_vector, dtype=np.float32)
        else:
            full_transaction = {**transaction, "account_history": history or []}
            try:
                features = extract_features(full_transaction)
            except Exception as fe:
                logger.error("Feature extraction failed: %s", fe)
                status = StreamStatus.DEGRADED
                features = np.zeros(16, dtype=np.float32)

        # Model inference
        result = predict(features)
        result["status"] = status if status == StreamStatus.DEGRADED else result["status"]

    except Exception as exc:
        logger.exception("Inference failed for transaction %s: %s", transaction_id, exc)
        return NumericalScoreResponse(
            transaction_id=transaction_id,
            score_numerical=0.5,   # neutral fallback — Fusion Layer will handle
            model_version=_model_version,
            status=StreamStatus.ERROR,
            signals=SignalScores(
                amount_anomaly=0.0,
                velocity_risk=0.0,
                pattern_deviation=0.0,
                geo_risk=0.0,
            ),
        )

    elapsed_ms = (time.monotonic() - start_time) * 1000
    if elapsed_ms > 50:
        logger.warning("Inference latency P95 breached: %.1f ms (target <= 50 ms)", elapsed_ms)

    return NumericalScoreResponse(
        transaction_id=transaction_id,
        score_numerical=result["score_numerical"],
        model_version=result["model_version"],
        status=result["status"],
        signals=SignalScores(
            amount_anomaly=result["amount_anomaly"],
            velocity_risk=result["velocity_risk"],
            pattern_deviation=result["pattern_deviation"],
            geo_risk=result["geo_risk"],
        ),
    )
