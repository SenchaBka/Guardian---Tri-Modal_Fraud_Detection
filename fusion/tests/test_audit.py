"""
Unit Tests – fusion.audit
===========================
Covers record, get, list_all, clear, export_jsonl.
"""

import json
import pytest
from fusion import audit as audit_store
from fusion.audit import AuditRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(**overrides):
    base = dict(
        request_id="req-001",
        transaction_id="TXN-001",
        decision="block",
        fraud_score=0.82,
        confidence=0.74,
        risk_level="critical",
        ensemble_mode="full",
        modalities_requested=["numerical", "nlp", "voice"],
        modalities_used=["numerical", "nlp", "voice"],
        modalities_failed=[],
        processing_time_ms=38.5,
        model_version="2.0.0",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class TestRecord:

    def test_returns_audit_record(self):
        entry = audit_store.record(**_sample())
        assert isinstance(entry, AuditRecord)

    def test_fields_stored_correctly(self):
        audit_store.record(**_sample())
        stored = audit_store.get("req-001")
        assert stored["transaction_id"] == "TXN-001"
        assert stored["decision"]       == "block"
        assert abs(stored["fraud_score"] - 0.82) < 1e-9

    def test_timestamp_is_iso_string(self):
        audit_store.record(**_sample())
        stored = audit_store.get("req-001")
        assert "T" in stored["timestamp"]   # ISO-8601 marker

    def test_override_applied_stored(self):
        audit_store.record(**_sample(override_applied=True, override_reason="Fallback override"))
        stored = audit_store.get("req-001")
        assert stored["override_applied"] is True
        assert "Fallback" in stored["override_reason"]

    def test_extra_metadata_stored(self):
        audit_store.record(**_sample(extra={"channel": "phone", "region": "CA"}))
        stored = audit_store.get("req-001")
        assert stored["extra"]["channel"] == "phone"

    def test_modalities_failed_stored(self):
        audit_store.record(**_sample(modalities_failed=["voice"]))
        stored = audit_store.get("req-001")
        assert stored["modalities_failed"] == ["voice"]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:

    def test_unknown_id_returns_none(self):
        assert audit_store.get("does-not-exist") is None

    def test_known_id_returns_dict(self):
        audit_store.record(**_sample())
        result = audit_store.get("req-001")
        assert isinstance(result, dict)

    def test_get_correct_entry_by_id(self):
        audit_store.record(**_sample(request_id="A", transaction_id="TXN-A"))
        audit_store.record(**_sample(request_id="B", transaction_id="TXN-B"))
        assert audit_store.get("A")["transaction_id"] == "TXN-A"
        assert audit_store.get("B")["transaction_id"] == "TXN-B"


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------

class TestListAll:

    def test_empty_store_returns_empty_list(self):
        assert audit_store.list_all() == []

    def test_single_record_listed(self):
        audit_store.record(**_sample())
        assert len(audit_store.list_all()) == 1

    def test_multiple_records_all_listed(self):
        for i in range(5):
            audit_store.record(**_sample(request_id=f"req-{i}", transaction_id=f"TXN-{i}"))
        assert len(audit_store.list_all()) == 5

    def test_returns_list_type(self):
        audit_store.record(**_sample())
        assert isinstance(audit_store.list_all(), list)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:

    def test_clear_empties_store(self):
        for i in range(3):
            audit_store.record(**_sample(request_id=f"req-{i}", transaction_id=f"TXN-{i}"))
        audit_store.clear()
        assert audit_store.list_all() == []

    def test_clear_twice_is_safe(self):
        audit_store.clear()
        audit_store.clear()
        assert audit_store.list_all() == []

    def test_can_add_after_clear(self):
        audit_store.record(**_sample())
        audit_store.clear()
        audit_store.record(**_sample(request_id="req-fresh", transaction_id="TXN-fresh"))
        assert len(audit_store.list_all()) == 1


# ---------------------------------------------------------------------------
# export_jsonl
# ---------------------------------------------------------------------------

class TestExportJsonl:

    def test_creates_file(self, tmp_path):
        audit_store.record(**_sample())
        out = tmp_path / "audit.jsonl"
        count = audit_store.export_jsonl(out)
        assert count == 1
        assert out.exists()

    def test_count_matches_records(self, tmp_path):
        for i in range(4):
            audit_store.record(**_sample(request_id=f"req-{i}", transaction_id=f"TXN-{i}"))
        out = tmp_path / "audit.jsonl"
        assert audit_store.export_jsonl(out) == 4

    def test_each_line_is_valid_json(self, tmp_path):
        for i in range(3):
            audit_store.record(**_sample(request_id=f"req-{i}", transaction_id=f"TXN-{i}"))
        out = tmp_path / "audit.jsonl"
        audit_store.export_jsonl(out)
        for line in out.read_text().strip().splitlines():
            parsed = json.loads(line)
            assert "request_id"     in parsed
            assert "transaction_id" in parsed
            assert "decision"       in parsed

    def test_empty_store_writes_empty_file(self, tmp_path):
        out = tmp_path / "empty.jsonl"
        count = audit_store.export_jsonl(out)
        assert count == 0
        assert out.read_text() == ""

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "audit.jsonl"
        audit_store.record(**_sample())
        audit_store.export_jsonl(out)
        assert out.exists()