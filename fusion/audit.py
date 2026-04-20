"""
Guardian Fusion Layer – Audit Trail
=====================================
Stores every FusionResponse decision in an in-memory log keyed by
request_id (UUID).  Supports retrieval, listing, clearing (for tests),
and JSONL export for compliance use.


Author: Sherwayne (ML Systems Architect)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# In-memory store (swap for Redis / Postgres in production)
# ---------------------------------------------------------------------------
_AUDIT_STORE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Audit record dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    """Full decision record persisted for every evaluated transaction."""
    request_id:            str
    transaction_id:        str
    timestamp:             str           # ISO-8601 UTC
    decision:              str           # approve / review / block
    fraud_score:           float
    confidence:            float
    risk_level:            str
    ensemble_mode:         str
    modalities_requested:  List[str]
    modalities_used:       List[str]
    modalities_failed:     List[str]
    processing_time_ms:    float
    model_version:         str
    override_applied:      bool          = False
    override_reason:       Optional[str] = None
    extra:                 Dict[str, Any]= field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record(
    *,
    request_id:           str,
    transaction_id:       str,
    decision:             str,
    fraud_score:          float,
    confidence:           float,
    risk_level:           str,
    ensemble_mode:        str,
    modalities_requested: List[str],
    modalities_used:      List[str],
    modalities_failed:    List[str],
    processing_time_ms:   float,
    model_version:        str,
    override_applied:     bool = False,
    override_reason:      Optional[str] = None,
    extra:                Optional[Dict[str, Any]] = None,
) -> AuditRecord:
    """
    Persist an audit record and return it.

    Parameters mirror the fields populated by FusionOrchestrator.evaluate()
    so the caller just passes what it already has.

    Returns
    -------
    AuditRecord  (also stored in _AUDIT_STORE under request_id)
    """
    entry = AuditRecord(
        request_id=request_id,
        transaction_id=transaction_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        fraud_score=fraud_score,
        confidence=confidence,
        risk_level=risk_level,
        ensemble_mode=ensemble_mode,
        modalities_requested=modalities_requested,
        modalities_used=modalities_used,
        modalities_failed=modalities_failed,
        processing_time_ms=processing_time_ms,
        model_version=model_version,
        override_applied=override_applied,
        override_reason=override_reason,
        extra=extra or {},
    )
    _AUDIT_STORE[request_id] = asdict(entry)
    return entry


def get(request_id: str) -> Optional[Dict[str, Any]]:
    """Return a stored audit record by request_id, or None if not found."""
    return _AUDIT_STORE.get(request_id)


def list_all() -> List[Dict[str, Any]]:
    """Return all stored audit records (insertion order)."""
    return list(_AUDIT_STORE.values())


def clear() -> None:
    """Wipe the audit store – intended for unit-test isolation only."""
    _AUDIT_STORE.clear()


def export_jsonl(path: str | Path) -> int:
    """
    Export all records to a JSON-Lines file for compliance archiving.

    Returns
    -------
    int : Number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record_dict in _AUDIT_STORE.values():
            fh.write(json.dumps(record_dict, default=str) + "\n")
            count += 1
    return count