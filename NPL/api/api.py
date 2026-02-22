"""guardian.nlp.api

FastAPI entrypoint for the NLP Stream.

Endpoint (Iteration #1 MVP):
- POST /api/v1/nlp/score

Input:
- RawNLPRequest (Fusion -> NLP)

Processing:
- RawNLPRequest -> NLPInput (canonicalization + cleaning)
- Semantic risk scoring (FinBERT if available, otherwise heuristic fallback)
- (Optional) placeholders for other signals (typosquatting, entity inconsistency)

Output:
- NLPOutput (NLP -> Fusion)

Run locally:
    uvicorn NPL.api.api:app --reload

Note:
- The Fusion layer should map its own request schema into RawNLPRequest when calling this service.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, HTTPException

from ..classifier import get_semantic_risk
from ..model_loader import healthcheck_model
from ..preprocessor import DEFAULT_CFG, raw_request_to_nlp_input
from .schemas import NLPOutput, NLPStatus, RawNLPRequest


app = FastAPI(
    title="Guardian NLP Stream",
    version="0.1.0",
    description="NLP modality service for the Guardian tri-modal fraud detection system.",
)


@app.get("/health")
def health():
    ok, msg = healthcheck_model()
    return {
        "status": "ok" if ok else "degraded",
        "model": msg,
    }


@app.post("/api/v1/nlp/score", response_model=NLPOutput)
def score_nlp(req: RawNLPRequest) -> NLPOutput:
    """Score text risk for a transaction."""

    t0 = time.time()

    try:
        nlp_input = raw_request_to_nlp_input(req, cfg=DEFAULT_CFG, combine_sources=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Semantic risk (primary MVP signal)
    sem = get_semantic_risk(nlp_input.text)

    # Placeholders (Iteration #1)
    typosquatting_risk: Optional[float] = None
    entity_inconsistency: Optional[float] = None

    # Determine status
    status = NLPStatus.ok
    if "[NO_TEXT]" in nlp_input.text or len(nlp_input.text.strip()) < 8:
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
        },
    )

    # (Optional) For debugging latency locally.
    # Fusion layer typically measures service latency in orchestration.
    _latency_ms = int((time.time() - t0) * 1000)

    return out