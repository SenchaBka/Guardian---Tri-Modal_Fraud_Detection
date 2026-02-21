"""
schemas.py
----------
Input/Output Pydantic contracts for the Guardian Numerical Stream API.
Consumed by: FastAPI routes (api.py) and the Fusion/Decision Layer.

Output contract is defined in: numerical_stream_contract.docx §5
Input (request) contract is defined in: numerical_stream_contract.docx §6
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransactionChannel(str, Enum):
    ONLINE = "online"
    PHONE = "phone"
    IN_PERSON = "in_person"
    ATM = "atm"


class StreamStatus(str, Enum):
    """
    ok        – Full inference completed with all features available.
    degraded  – One or more features missing; score still produced.
                Fusion Layer applies +5 % confidence penalty.
    error     – Inference failed; no score produced.
                Fusion Layer excludes Numerical and triggers human review.
    """
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TransactionData(BaseModel):
    """Core transaction fields required for feature engineering."""

    amount: float = Field(..., description="Transaction value in base currency unit", gt=0)
    currency: str = Field(..., description="ISO 4217 currency code", min_length=3, max_length=3)
    timestamp: datetime = Field(..., description="UTC ISO 8601 transaction time")
    channel: Optional[TransactionChannel] = Field(None, description="Transaction channel")
    country: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 country code", min_length=2, max_length=2)
    merchant_category: Optional[str] = Field(None, description="MCC or merchant category label")


class NumericalFeatures(BaseModel):
    """
    Optional pre-computed feature data.
    If feature_vector is supplied it bypasses on-the-fly feature engineering.
    raw_features carries account/history data used for velocity and z-score computation.
    """

    feature_vector: Optional[List[float]] = Field(
        None, description="Pre-computed numerical feature array (optional override)"
    )
    raw_features: Optional[dict] = Field(
        None, description="Raw account/history data for on-the-fly feature extraction"
    )


class NumericalScoreRequest(BaseModel):
    """
    POST /api/v1/numerical/score  — request body.
    Defined in numerical_stream_contract.docx §6.
    """

    transaction_id: str = Field(..., description="Unique transaction identifier")
    transaction_data: TransactionData
    numerical_features: Optional[NumericalFeatures] = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SignalScores(BaseModel):
    """Four interpretable sub-signals returned alongside the primary score."""

    amount_anomaly: float = Field(
        ..., ge=0.0, le=1.0,
        description="Z-score-derived risk: transaction amount vs 90-day account history"
    )
    velocity_risk: float = Field(
        ..., ge=0.0, le=1.0,
        description="Rolling-window (1h/24h/7d) transaction frequency anomaly risk"
    )
    pattern_deviation: float = Field(
        ..., ge=0.0, le=1.0,
        description="Isolation Forest anomaly score over (amount, time-of-day, channel, MCC)"
    )
    geo_risk: float = Field(
        ..., ge=0.0, le=1.0,
        description="Composite geographical risk: country tier + travel distance + impossible-travel flag"
    )


class NumericalScoreResponse(BaseModel):
    """
    POST /api/v1/numerical/score  — response body.
    Defined in numerical_stream_contract.docx §5.
    This is the contract consumed by the Fusion/Decision Layer (Sherwayne).
    """

    transaction_id: str = Field(..., description="Echoes the transaction_id from the request")
    score_numerical: float = Field(
        ..., ge=0.0, le=1.0,
        description="Calibrated fraud probability [0,1] — primary value for Fusion Layer"
    )
    model_version: str = Field(..., description="Semver identifier of the deployed model artefact")
    status: StreamStatus = Field(..., description="Inference status: ok | degraded | error")
    signals: SignalScores = Field(..., description="Four granular sub-signal scores for Fusion explainability")


# ---------------------------------------------------------------------------
# Health & model-info schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """GET /api/v1/numerical/health"""

    status: str = Field(..., description="'healthy' or 'unhealthy'")
    model_loaded: bool
    feature_store_available: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """GET /api/v1/numerical/model"""

    model_version: str
    training_date: Optional[str] = None
    algorithm: str = "XGBoost + Platt Scaling"
    metrics: Optional[dict] = Field(
        None,
        description="Snapshot of evaluation metrics: precision, recall, f1, auc_roc, fpr"
    )
