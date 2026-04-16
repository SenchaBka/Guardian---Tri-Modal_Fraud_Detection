"""guardian.nlp.api

FastAPI entrypoint for the NLP Stream.

Endpoint (Iteration 2):
- POST /api/v1/nlp/score

Input:
- RawNLPRequest (Fusion -> NLP)

Processing:
- RawNLPRequest -> NLPInput (canonicalization + cleaning)
- Fraud probability scoring using the fine-tuned PaySim FinBERT model
- Heuristic fallback only if the local model cannot load

Output:
- NLPOutput (NLP -> Fusion)

Run locally:
    uvicorn NPL.api.api:app --reload

Note:
- The Fusion layer should map its own request schema into RawNLPRequest when calling this service.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException

from ..classifier import get_semantic_risk
from ..model_loader import healthcheck_model, load_trained_nlp_bundle
from ..preprocessor import DEFAULT_CFG, raw_request_to_nlp_input
from .schemas import NLPOutput, NLPStatus, RawNLPRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Guardian NLP Stream",
    version="0.1.0",
    description="NLP modality service for the Guardian tri-modal fraud detection system.",
)


@app.on_event("startup")
def warmup_model() -> None:
    """Warm-load the trained NLP model when the API starts."""
    try:
        logger.info("Warming up NLP model on API startup")
        load_trained_nlp_bundle()
    except Exception:
        # Keep API bootable even if model load fails; health endpoint will report it.
        logger.error("Failed to warm up NLP model during startup", exc_info=True)
        pass


@app.get("/health")
def health():
    ok, msg = healthcheck_model()
    response = {
        "status": "ok" if ok else "degraded",
        "model": msg,
    }

    try:
        bundle = load_trained_nlp_bundle()
        response["threshold"] = float(bundle.threshold)
        response["model_version"] = str(bundle.model_name)
        response["device"] = str(bundle.device)
        response["model_source"] = str(getattr(bundle, "source", "unknown"))
        response["model_revision"] = str(getattr(bundle, "revision", "unknown"))
    except Exception:
        pass

    return response


@app.post("/api/v1/nlp/score", response_model=NLPOutput)
def score_nlp(req: RawNLPRequest) -> NLPOutput:
    """Score fraud risk for a transaction using the trained NLP model."""

    t0 = time.time()
    logger.info("Incoming request: %s", req.transaction_id)

    try:
        nlp_input = raw_request_to_nlp_input(req, cfg=DEFAULT_CFG, combine_sources=True)
        logger.info("Text length: %s", len(nlp_input.text))
    except Exception as e:
        logger.error("Error processing request", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))

    # Fraud probability from the trained transformer classifier
    sem = get_semantic_risk(nlp_input.text)
    logger.info(
        "Score: %s | Threshold: %s",
        float(sem.score),
        float(sem.threshold) if sem.threshold is not None else "none",
    )

    # Placeholders (Iteration #1)
    typosquatting_risk: Optional[float] = None
    entity_inconsistency: Optional[float] = None

    # Determine status
    status = NLPStatus.ok
    if "[NO_TEXT]" in nlp_input.text or len(nlp_input.text.strip()) < 8:
        status = NLPStatus.degraded
    elif str(sem.model_version) == "heuristic_v1":
        status = NLPStatus.degraded

    # Compose output
    out = NLPOutput(
        transaction_id=nlp_input.transaction_id,
        score_nlp=float(sem.score),
        model_version=str(sem.model_version),
        status=status,
        signals={
            "semantic_risk": float(sem.score),
            "typosquatting_risk": typosquatting_risk,
            "entity_inconsistency": entity_inconsistency,
            "threshold_used": float(sem.threshold) if sem.threshold is not None else None,
            "predicted_fraud": (
                float(sem.details.get("predicted_fraud"))
                if sem.details and "predicted_fraud" in sem.details
                else None
            ),
        },
    )

    # (Optional) For debugging latency locally.
    # Fusion layer typically measures service latency in orchestration.
    _latency_ms = int((time.time() - t0) * 1000)

    # Useful local debug values (not part of the external contract):
    # - fraud_probability = out.score_nlp
    # - threshold_used = out.signals.get("threshold_used")
    # - predicted_fraud = out.signals.get("predicted_fraud")
    logger.info(
        "Completed request %s with status=%s in %sms",
        req.transaction_id,
        out.status.value,
        _latency_ms,
    )

    return out
