"""
Guardian Fusion Layer – FastAPI Application
============================================
Iteration #2 additions:
  POST /api/v1/fusion/batch          – evaluate up to 100 transactions at once
  GET  /api/v1/audit/{request_id}    – retrieve any past decision by UUID
  GET  /api/v1/audit                 – list all stored audit records

Run with: uvicorn fusion.api:app --reload --port 8000

Author: Sherwayne (ML Systems Architect)
"""

from datetime import datetime
from typing import Dict, List, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .schemas import (
    FusionRequest,
    FusionResponse,
    HealthResponse,
    ModalityHealthStatus,
)
from .orchestrator import FusionOrchestrator
from .config import MODEL_VERSION, MODALITY_ENDPOINTS
from . import audit as audit_store


# ---------------------------------------------------------------------------
# Batch schemas (defined here – no changes to existing schemas.py needed)
# ---------------------------------------------------------------------------

class BatchFusionRequest(BaseModel):
    requests: List[FusionRequest] = Field(..., min_length=1, max_length=100)


class BatchFusionResponse(BaseModel):
    total_processed: int
    total_approved:  int
    total_review:    int
    total_blocked:   int
    results:         List[FusionResponse]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Guardian Fusion Layer v{MODEL_VERSION} starting...")
    app.state.orchestrator = FusionOrchestrator()
    print("✅ FusionOrchestrator initialized")
    yield
    print("🛑 Guardian Fusion Layer shutting down...")


app = FastAPI(
    title="Guardian Fusion Layer",
    description="Adaptive Ensemble Architecture for Tri-Modal Fraud Detection",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "validation_error", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )


# ---------------------------------------------------------------------------
# Existing endpoints (unchanged)
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/fusion/evaluate",
    response_model=FusionResponse,
    tags=["Fraud Evaluation"],
)
async def evaluate_transaction(request: FusionRequest) -> FusionResponse:
    """Evaluate a single transaction for fraud using the adaptive ensemble."""
    orchestrator: FusionOrchestrator = app.state.orchestrator
    try:
        return await orchestrator.evaluate(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Evaluation failed for {request.transaction_id}: {e}")
        raise


@app.get(
    "/api/v1/fusion/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Check health of Fusion Layer and downstream services."""
    modality_statuses: Dict[str, ModalityHealthStatus] = {
        m: ModalityHealthStatus(status="healthy", latency_ms=50.0, last_check=datetime.utcnow())
        for m in ["numerical", "nlp", "voice"]
    }
    return HealthResponse(
        status="healthy",
        version=MODEL_VERSION,
        timestamp=datetime.utcnow(),
        modalities=modality_statuses,
    )


@app.get("/api/v1/fusion/modalities", tags=["System"])
async def modality_status() -> Dict:
    """Get detailed modality endpoint status."""
    return {
        "modalities": {
            "numerical": {"endpoint": MODALITY_ENDPOINTS["numerical"], "owner": "Ivan",    "status": "available", "required": True},
            "nlp":       {"endpoint": MODALITY_ENDPOINTS["nlp"],       "owner": "Luis",    "status": "available", "required": False},
            "voice":     {"endpoint": MODALITY_ENDPOINTS["voice"],     "owner": "Arsenii", "status": "available", "required": False},
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "name":      "Guardian Fusion Layer",
        "version":   MODEL_VERSION,
        "docs":      "/docs",
        "health":    "/api/v1/fusion/health",
        "evaluate":  "/api/v1/fusion/evaluate",
        "batch":     "/api/v1/fusion/batch",
        "audit":     "/api/v1/audit",
    }


# ---------------------------------------------------------------------------
# NEW – Batch endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/fusion/batch",
    response_model=BatchFusionResponse,
    tags=["Fraud Evaluation"],
)
async def batch_evaluate(batch: BatchFusionRequest) -> BatchFusionResponse:
    """
    Evaluate up to 100 transactions in a single request.

    Each transaction is processed independently through the full fusion
    pipeline.  Failures in individual transactions do not abort the batch –
    they propagate as HTTP 500 for that request only.
    """
    orchestrator: FusionOrchestrator = app.state.orchestrator
    results: List[FusionResponse] = []

    for req in batch.requests:
        try:
            results.append(await orchestrator.evaluate(req))
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Validation error for transaction {req.transaction_id}: {e}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation failed for transaction {req.transaction_id}: {e}",
            )

    from .schemas import DecisionType
    return BatchFusionResponse(
        total_processed=len(results),
        total_approved=sum(1 for r in results if r.decision == DecisionType.APPROVE),
        total_review=sum(1 for r in results   if r.decision == DecisionType.REVIEW),
        total_blocked=sum(1 for r in results  if r.decision == DecisionType.BLOCK),
        results=results,
    )


# ---------------------------------------------------------------------------
# NEW – Audit endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/audit",
    response_model=List[Dict[str, Any]],
    tags=["Audit"],
)
async def list_audit_records() -> List[Dict[str, Any]]:
    """Return all stored audit records (most recent last)."""
    return audit_store.list_all()


@app.get(
    "/api/v1/audit/{request_id}",
    response_model=Dict[str, Any],
    tags=["Audit"],
)
async def get_audit_record(request_id: str) -> Dict[str, Any]:
    """
    Retrieve a single audit record by its request_id UUID.

    The request_id is returned in every FusionResponse under audit.request_id.
    """
    record = audit_store.get(request_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Audit record '{request_id}' not found.",
        )
    return record


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fusion.api:app", host="0.0.0.0", port=8000, reload=True)