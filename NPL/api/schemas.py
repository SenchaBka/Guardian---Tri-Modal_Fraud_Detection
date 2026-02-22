"""guardian.nlp.schemas

Pydantic schemas for the NLP Stream.

This module defines TWO levels of input:
1) RawNLPRequest: what the Orchestration/Fusion layer sends at runtime.
2) NLPInput: the canonical internal contract used by the NLP pipeline (training + inference).

And ONE output:
- NLPOutput: what the NLP Stream returns to the Fusion layer.

Design goals:
- Contract-driven (stable schema regardless of dataset source)
- Graceful degradation (missing optional fields do not break inference)
- Explicit signals for fusion/explainability
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

try:
    # Pydantic v2
    from pydantic import ConfigDict
    from pydantic import field_validator, model_validator

    PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    # Pydantic v1 fallback
    from pydantic import validator as field_validator  # type: ignore
    from pydantic import root_validator as model_validator  # type: ignore

    class ConfigDict(dict):  # type: ignore
        pass

    PYDANTIC_V2 = False


# -----------------------------
# Enums / Literals
# -----------------------------


class NLPStatus(str, Enum):
    """Runtime status returned by the NLP Stream."""

    ok = "ok"
    degraded = "degraded"  # missing text fields, short text, etc.
    error = "error"  # unexpected failure


TextSource = Literal["merchant", "narrative", "invoice", "ticket", "combined"]


# -----------------------------
# Shared sub-schemas
# -----------------------------


class NLPMetadata(BaseModel):
    """Optional transaction metadata.

    Keep this lightweight and non-blocking. Missing values are expected.
    """

    amount: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = None
    country: Optional[str] = None
    channel: Optional[str] = None
    transaction_type: Optional[str] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")

    else:
        class Config:
            extra = "allow"


class NLPTextPayload(BaseModel):
    """Raw text payload sent by the Orchestration/Fusion layer."""

    merchant_text: Optional[str] = None
    narrative_text: Optional[str] = None
    invoice_text: Optional[str] = None
    ticket_text: Optional[str] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    else:
        class Config:
            extra = "forbid"


# -----------------------------
# Input contracts
# -----------------------------


class RawNLPRequest(BaseModel):
    """Runtime input contract (Sherwayne/Fusion -> NLP API).

    Rules:
    - transaction_id required
    - payload must contain at least one non-empty text field
    - metadata optional
    - language optional (defaults to 'en')
    """

    transaction_id: str = Field(..., min_length=1)
    language: str = Field(default="en", min_length=2, max_length=12)
    payload: NLPTextPayload
    metadata: NLPMetadata = Field(default_factory=NLPMetadata)

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

        @model_validator(mode="after")
        def _validate_payload_has_text(self):
            texts = [
                self.payload.merchant_text,
                self.payload.narrative_text,
                self.payload.invoice_text,
                self.payload.ticket_text,
            ]
            has_text = any((t is not None and str(t).strip() != "") for t in texts)
            if not has_text:
                raise ValueError("payload must include at least one non-empty text field")
            return self

    else:
        class Config:
            extra = "forbid"

        @model_validator
        def _validate_payload_has_text(cls, values: Dict[str, Any]):  # type: ignore
            payload = values.get("payload")
            if payload is None:
                raise ValueError("payload is required")
            texts = [
                getattr(payload, "merchant_text", None),
                getattr(payload, "narrative_text", None),
                getattr(payload, "invoice_text", None),
                getattr(payload, "ticket_text", None),
            ]
            has_text = any((t is not None and str(t).strip() != "") for t in texts)
            if not has_text:
                raise ValueError("payload must include at least one non-empty text field")
            return values


class NLPInput(BaseModel):
    """Canonical internal contract used by the NLP pipeline.

    NOTE: This is NOT exposed to external callers. It is produced by the NLP service
    after validation + cleaning.
    """

    transaction_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    text_source: TextSource = Field(default="combined")
    language: str = Field(default="en", min_length=2, max_length=12)
    metadata: NLPMetadata = Field(default_factory=NLPMetadata)

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

        @field_validator("text")
        @classmethod
        def _strip_text(cls, v: str) -> str:
            v2 = v.strip()
            if not v2:
                raise ValueError("text cannot be empty")
            return v2

    else:
        class Config:
            extra = "forbid"

        @field_validator("text")
        def _strip_text(cls, v: str) -> str:  # type: ignore
            v2 = v.strip()
            if not v2:
                raise ValueError("text cannot be empty")
            return v2


# -----------------------------
# Output contract
# -----------------------------


class NLPSignals(BaseModel):
    """Optional sub-signals to support explainability + fusion."""

    semantic_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    typosquatting_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    entity_inconsistency: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")

    else:
        class Config:
            extra = "allow"


class NLPOutput(BaseModel):
    """NLP Stream response contract (NLP -> Fusion).

    Guarantees:
    - score_nlp always in [0, 1]
    - transaction_id echoed back
    - signals are optional and may be partially populated
    """

    transaction_id: str = Field(..., min_length=1)
    score_nlp: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(..., min_length=1)
    status: NLPStatus = Field(default=NLPStatus.ok)
    signals: NLPSignals = Field(default_factory=NLPSignals)

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    else:
        class Config:
            extra = "forbid"