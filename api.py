"""
api.py
------
FastAPI routes for the Guardian Numerical Stream.

Endpoints (numerical_stream_contract.docx §8):
  POST /api/v1/numerical/score   — full transaction scoring
  GET  /api/v1/numerical/health  — service health check
  GET  /api/v1/numerical/model   — deployed model info

SLA (numerical_stream_contract.docx §9):
  P50 < 30 ms  |  P95 < 50 ms  |  Availability 99.9 %  |  > 150 TPS
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from inference import is_model_loaded, load_models, score_transaction
from models import MODEL_VERSION
from schemas import (
    HealthResponse,
    ModelInfoResponse,
    NumericalScoreRequest,
    NumericalScoreResponse,
    StreamStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App startup / shutdown
# ---------------------------------------------------------------------------

_start_time: float = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once on startup."""
    logger.info("Guardian Numerical Stream — loading models...")
    load_models()
    logger.info("Models ready. Service is up.")
    yield
    logger.info("Guardian Numerical Stream — shutting down.")


app = FastAPI(
    title="Guardian Numerical Stream API",
    description=(
        "Transactional intelligence layer of the Guardian Tri-Modal Fraud Detection System. "
        "Produces score_numerical ∈ [0,1] and four interpretable sub-signals for the Fusion Layer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

router = APIRouter(prefix="/api/v1/numerical", tags=["numerical"])


# ---------------------------------------------------------------------------
# POST /api/v1/numerical/score
# ---------------------------------------------------------------------------

@router.post(
    "/score",
    response_model=NumericalScoreResponse,
    summary="Score a transaction",
    description=(
        "Full transaction scoring with on-the-fly feature engineering. "
        "Returns score_numerical and all four sub-signals (amount_anomaly, "
        "velocity_risk, pattern_deviation, geo_risk). "
        "This is the primary endpoint consumed by the Fusion/Decision Layer."
    ),
)
async def score(request: NumericalScoreRequest) -> NumericalScoreResponse:
    """
    Accept a raw transaction and return a calibrated fraud probability.

    If `numerical_features.feature_vector` is provided it bypasses
    on-the-fly feature engineering (useful for batch or pre-computed scenarios).
    """
    # Unpack optional pre-computed features
    feature_vector = None
    history = None

    if request.numerical_features:
        if request.numerical_features.feature_vector:
            feature_vector = request.numerical_features.feature_vector
        if request.numerical_features.raw_features:
            history = request.numerical_features.raw_features.get("transaction_history", [])

    transaction_dict = request.transaction_data.model_dump()

    response = score_transaction(
        transaction_id=request.transaction_id,
        transaction=transaction_dict,
        history=history,
        feature_vector=feature_vector,
    )

    # If inference errored, return 200 with status=error per the API contract.
    # The Fusion Layer handles the error status by triggering human review.
    return response


# ---------------------------------------------------------------------------
# GET /api/v1/numerical/health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health() -> HealthResponse:
    """
    Returns operational status of the Numerical Stream service.
    The Fusion Layer polls this endpoint to decide whether to include
    Numerical in the ensemble.
    """
    uptime = time.monotonic() - _start_time
    model_loaded = is_model_loaded()

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        feature_store_available=True,   # update when a real feature store is wired
        uptime_seconds=round(uptime, 2),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/numerical/model
# ---------------------------------------------------------------------------

@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Deployed model information",
)
async def model_info() -> ModelInfoResponse:
    """
    Returns the currently deployed model version, algorithm, and a snapshot
    of the evaluation metrics from the most recent training run.

    Metric targets (numerical_stream_contract.docx §7):
      Precision  >= 0.85 | Recall >= 0.80 | F1 >= 0.82 | AUC-ROC >= 0.95 | FPR <= 0.02
    """
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        training_date=None,   # populated after first real training run
        algorithm="XGBoost + Platt Scaling",
        metrics={
            "precision_target": 0.85,
            "recall_target": 0.80,
            "f1_target": 0.82,
            "auc_roc_target": 0.95,
            "fpr_target": 0.02,
        },
    )


# ---------------------------------------------------------------------------
# Register router
# ---------------------------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
