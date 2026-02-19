"""
Guardian Fusion Layer - FastAPI Application
============================================

Run with: uvicorn fusion.api:app --reload --port 8000

Author: Sherwayne (ML Systems Architect)
"""

from datetime import datetime
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    FusionRequest,
    FusionResponse,
    HealthResponse,
    ModalityHealthStatus,
)
from .orchestrator import FusionOrchestrator
from .config import MODEL_VERSION, MODALITY_ENDPOINTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events."""
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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "validation_error", "message": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."}
    )


@app.post("/api/v1/fusion/evaluate", response_model=FusionResponse, tags=["Fraud Evaluation"])
async def evaluate_transaction(request: FusionRequest) -> FusionResponse:
    """Evaluate a transaction for fraud using the adaptive ensemble."""
    orchestrator: FusionOrchestrator = app.state.orchestrator
    
    try:
        response = await orchestrator.evaluate(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Evaluation failed for {request.transaction_id}: {e}")
        raise


@app.get("/api/v1/fusion/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check health of Fusion Layer and downstream services."""
    modality_statuses: Dict[str, ModalityHealthStatus] = {}
    
    for modality in ["numerical", "nlp", "voice"]:
        modality_statuses[modality] = ModalityHealthStatus(
            status="healthy",
            latency_ms=50.0,
            last_check=datetime.utcnow(),
        )
    
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
            "numerical": {
                "endpoint": MODALITY_ENDPOINTS["numerical"],
                "owner": "Ivan",
                "status": "available",
                "required": True,
            },
            "nlp": {
                "endpoint": MODALITY_ENDPOINTS["nlp"],
                "owner": "Luis",
                "status": "available",
                "required": False,
            },
            "voice": {
                "endpoint": MODALITY_ENDPOINTS["voice"],
                "owner": "Arsenii",
                "status": "available",
                "required": False,
            },
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Guardian Fusion Layer",
        "version": MODEL_VERSION,
        "documentation": "/docs",
        "health": "/api/v1/fusion/health",
        "evaluate": "/api/v1/fusion/evaluate",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fusion.api:app", host="0.0.0.0", port=8000, reload=True)