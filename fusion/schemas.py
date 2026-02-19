"""
Guardian Fusion Layer - Pydantic Schema Definitions
====================================================

Author: Sherwayne (ML Systems Architect)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class DecisionType(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EnsembleMode(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    FALLBACK = "fallback"


class ModalityStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class ChannelType(str, Enum):
    ONLINE = "online"
    PHONE = "phone"
    IN_PERSON = "in_person"
    ATM = "atm"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class AvailableModalities(BaseModel):
    nlp: bool = Field(default=True)
    numerical: bool = Field(default=True)
    voice: bool = Field(default=False)
    
    @field_validator('numerical')
    @classmethod
    def numerical_must_be_true(cls, v: bool) -> bool:
        if not v:
            return True
        return v


class TransactionData(BaseModel):
    amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    channel: ChannelType = Field(default=ChannelType.ONLINE)
    country: Optional[str] = Field(default=None)
    merchant_category: Optional[str] = Field(default=None)


class TextPayload(BaseModel):
    merchant_text: Optional[str] = Field(default=None)
    narrative_text: Optional[str] = Field(default=None)
    invoice_text: Optional[str] = Field(default=None)


class VoicePayload(BaseModel):
    audio_url: Optional[str] = Field(default=None)
    audio_base64: Optional[str] = Field(default=None)
    sample_rate: int = Field(default=16000)


class NumericalFeatures(BaseModel):
    feature_vector: Optional[List[float]] = Field(default=None)
    raw_features: Optional[Dict[str, Any]] = Field(default=None)


class FusionOptions(BaseModel):
    explain: bool = Field(default=True)
    threshold_override: Optional[float] = Field(default=None, ge=0, le=1)
    force_review_on_fallback: bool = Field(default=True)


class FusionRequest(BaseModel):
    transaction_id: str = Field(...)
    available_modalities: Optional[AvailableModalities] = Field(default=None)
    transaction_data: TransactionData = Field(...)
    text_payload: Optional[TextPayload] = Field(default=None)
    voice_payload: Optional[VoicePayload] = Field(default=None)
    numerical_features: Optional[NumericalFeatures] = Field(default=None)
    options: FusionOptions = Field(default_factory=FusionOptions)


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class ModalityScoreDetail(BaseModel):
    score: Optional[float] = Field(default=None)
    status: ModalityStatus = Field(...)
    weight: float = Field(...)
    used: bool = Field(...)


class ModalityScores(BaseModel):
    nlp: ModalityScoreDetail = Field(...)
    numerical: ModalityScoreDetail = Field(...)
    voice: ModalityScoreDetail = Field(...)


class ConfidenceAdjustment(BaseModel):
    base_confidence: float = Field(...)
    availability_penalty: float = Field(...)
    final_confidence: float = Field(...)


class TopFactor(BaseModel):
    feature: str = Field(...)
    impact: float = Field(...)
    direction: Literal["positive", "negative"] = Field(...)


class ShapValues(BaseModel):
    nlp_contribution: Optional[float] = Field(default=None)
    numerical_contribution: float = Field(...)
    voice_contribution: Optional[float] = Field(default=None)


class Explanation(BaseModel):
    top_factors: List[TopFactor] = Field(...)
    shap_values: ShapValues = Field(...)
    narrative: str = Field(...)
    missing_modality_note: Optional[str] = Field(default=None)


class AuditInfo(BaseModel):
    request_id: str = Field(...)
    processing_time_ms: float = Field(...)
    modalities_requested: List[str] = Field(...)
    modalities_used: List[str] = Field(...)
    modalities_failed: List[str] = Field(default_factory=list)


class FusionResponse(BaseModel):
    transaction_id: str = Field(...)
    fraud_score: float = Field(..., ge=0, le=1)
    decision: DecisionType = Field(...)
    confidence: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel = Field(...)
    ensemble_mode: EnsembleMode = Field(...)
    model_version: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    modality_scores: ModalityScores = Field(...)
    confidence_adjustment: ConfidenceAdjustment = Field(...)
    explanation: Explanation = Field(...)
    audit: AuditInfo = Field(...)


# =============================================================================
# HEALTH CHECK SCHEMAS
# =============================================================================

class ModalityHealthStatus(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(...)
    latency_ms: Optional[float] = Field(default=None)
    last_check: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(...)
    version: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    modalities: Dict[str, ModalityHealthStatus] = Field(...)