"""
Unit / Integration Tests - fusion.api
All tests use a session-scoped TestClient that runs inside the FastAPI
lifespan context, so app.state.orchestrator is properly initialized.
random.uniform is patched per-test to make mock responses deterministic.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from fusion.api import app
from fusion import audit as audit_store

_R = [0.30, 0.25, 0.15, 0.20, 0.12, 0.25, 0.10] * 60

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def reset_audit():
    audit_store.clear()
    yield
    audit_store.clear()

def _txn(amount=250.0, channel="online", country="US", merchant_category="retail"):
    return {"amount": amount, "currency": "USD", "channel": channel, "country": country, "merchant_category": merchant_category}

def _online_request(txn_id="TXN-001", amount=250.0):
    return {
        "transaction_id": txn_id,
        "available_modalities": {"nlp": True, "voice": False},
        "transaction_data": _txn(amount=amount),
        "text_payload": {"merchant_text": "Amazon", "narrative_text": "online purchase"},
        "options": {"explain": True, "force_review_on_fallback": True},
    }

def _fallback_request(txn_id="TXN-FB"):
    return {
        "transaction_id": txn_id,
        "available_modalities": {"nlp": False, "voice": False},
        "transaction_data": _txn(),
        "options": {"explain": True, "force_review_on_fallback": True},
    }

def _phone_request(txn_id="TXN-PHONE"):
    return {
        "transaction_id": txn_id,
        "available_modalities": {"nlp": True, "voice": True},
        "transaction_data": _txn(channel="phone"),
        "text_payload": {"merchant_text": "merchant", "narrative_text": "phone purchase"},
        "voice_payload": {"audio_url": "http://example.com/a.wav"},
        "options": {"explain": True},
    }


class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/api/v1/fusion/health").status_code == 200

    def test_status_healthy(self, client):
        assert client.get("/api/v1/fusion/health").json()["status"] == "healthy"

    def test_version_present(self, client):
        assert "version" in client.get("/api/v1/fusion/health").json()

    def test_all_modalities_listed(self, client):
        assert set(client.get("/api/v1/fusion/health").json()["modalities"].keys()) == {"numerical", "nlp", "voice"}


class TestEvaluateEndpoint:
    def test_status_200(self, client):
        with patch("random.uniform", side_effect=_R):
            assert client.post("/api/v1/fusion/evaluate", json=_online_request()).status_code == 200

    def test_transaction_id_echoed(self, client):
        with patch("random.uniform", side_effect=_R):
            r = client.post("/api/v1/fusion/evaluate", json=_online_request("TXN-XYZ"))
        assert r.json()["transaction_id"] == "TXN-XYZ"

    def test_decision_valid_enum(self, client):
        with patch("random.uniform", side_effect=_R):
            r = client.post("/api/v1/fusion/evaluate", json=_online_request())
        assert r.json()["decision"] in ("approve", "review", "block")

    def test_fraud_score_in_unit_interval(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert 0.0 <= body["fraud_score"] <= 1.0

    def test_confidence_in_unit_interval(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_risk_level_valid(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert body["risk_level"] in ("low", "medium", "high", "critical")

    def test_partial_mode_when_nlp_only(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert body["ensemble_mode"] == "partial"

    def test_fallback_mode(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_fallback_request()).json()
        assert body["ensemble_mode"] == "fallback"

    def test_full_mode(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_phone_request()).json()
        assert body["ensemble_mode"] == "full"

    def test_audit_request_id_is_uuid(self, client):
        import re
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", body["audit"]["request_id"])

    def test_explanation_narrative_present(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert len(body["explanation"]["narrative"]) > 0

    def test_modality_scores_all_keys(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert set(body["modality_scores"].keys()) == {"numerical", "nlp", "voice"}

    def test_confidence_adjustment_fields(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        ca = body["confidence_adjustment"]
        for field in ("base_confidence", "availability_penalty", "final_confidence"):
            assert field in ca

    def test_fallback_higher_penalty_than_partial(self, client):
        with patch("random.uniform", side_effect=_R):
            partial  = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        with patch("random.uniform", side_effect=_R):
            fallback = client.post("/api/v1/fusion/evaluate", json=_fallback_request()).json()
        assert fallback["confidence_adjustment"]["availability_penalty"] > partial["confidence_adjustment"]["availability_penalty"]

    def test_model_version_present(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert body["model_version"] != ""

    def test_processing_time_positive(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        assert body["audit"]["processing_time_ms"] > 0

    def test_nlp_score_none_when_voice_only_partial(self, client):
        req = {
            "transaction_id": "TXN-VOICE",
            "available_modalities": {"nlp": False, "voice": True},
            "transaction_data": _txn(),
            "voice_payload": {"audio_url": "http://example.com/a.wav"},
            "options": {"explain": True},
        }
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=req).json()
        assert body["modality_scores"]["nlp"]["used"] is False


class TestFallbackOverrides:

    def test_high_risk_fallback_not_approved(self, client):
        req = {
            "transaction_id": "TXN-HIGH",
            "available_modalities": {"nlp": False, "voice": False},
            "transaction_data": _txn(amount=80_000.0, country="NG", merchant_category="crypto"),
            "options": {"explain": True, "force_review_on_fallback": True},
        }
        with patch("random.uniform", side_effect=[0.45, 0.75, 0.70] * 20):
            body = client.post("/api/v1/fusion/evaluate", json=req).json()
        assert body["decision"] in ("review", "block")

    def test_force_review_false_allows_approve_for_low_score(self, client):
        req = {
            "transaction_id": "TXN-NOOVERRIDE",
            "available_modalities": {"nlp": False, "voice": False},
            "transaction_data": _txn(amount=10.0, country="US"),
            "options": {"explain": False, "force_review_on_fallback": False},
        }
        with patch("random.uniform", side_effect=[0.05, 0.05, 0.05] * 20):
            body = client.post("/api/v1/fusion/evaluate", json=req).json()
        assert body["decision"] == "approve"


class TestEvaluateValidation:

    def test_missing_transaction_data_422(self, client):
        assert client.post("/api/v1/fusion/evaluate", json={"transaction_id": "TXN-X"}).status_code == 422

    def test_empty_transaction_id_422(self, client):
        req = _online_request()
        req["transaction_id"] = ""
        assert client.post("/api/v1/fusion/evaluate", json=req).status_code == 422

    def test_negative_amount_422(self, client):
        req = _online_request()
        req["transaction_data"]["amount"] = -100.0
        assert client.post("/api/v1/fusion/evaluate", json=req).status_code == 422

    def test_threshold_override_above_one_422(self, client):
        req = _online_request()
        req["options"]["threshold_override"] = 1.5
        assert client.post("/api/v1/fusion/evaluate", json=req).status_code == 422

    def test_threshold_override_below_zero_422(self, client):
        req = _online_request()
        req["options"]["threshold_override"] = -0.1
        assert client.post("/api/v1/fusion/evaluate", json=req).status_code == 422


class TestBatchEndpoint:

    def test_returns_200(self, client):
        payload = {"requests": [_online_request(f"TXN-{i}") for i in range(3)]}
        with patch("random.uniform", side_effect=_R):
            assert client.post("/api/v1/fusion/batch", json=payload).status_code == 200

    def test_total_processed_correct(self, client):
        payload = {"requests": [_online_request(f"TXN-{i}") for i in range(4)]}
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/batch", json=payload).json()
        assert body["total_processed"] == 4

    def test_results_count_matches_processed(self, client):
        payload = {"requests": [_online_request(f"TXN-{i}") for i in range(3)]}
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/batch", json=payload).json()
        assert len(body["results"]) == body["total_processed"]

    def test_totals_sum_to_processed(self, client):
        payload = {"requests": [_online_request(f"TXN-{i}") for i in range(5)]}
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/batch", json=payload).json()
        assert body["total_approved"] + body["total_review"] + body["total_blocked"] == body["total_processed"]

    def test_each_result_has_transaction_id(self, client):
        ids = [f"TXN-{i}" for i in range(3)]
        payload = {"requests": [_online_request(i) for i in ids]}
        with patch("random.uniform", side_effect=_R):
            results = client.post("/api/v1/fusion/batch", json=payload).json()["results"]
        assert set(r["transaction_id"] for r in results) == set(ids)

    def test_empty_batch_422(self, client):
        assert client.post("/api/v1/fusion/batch", json={"requests": []}).status_code == 422

    def test_mixed_modes_in_batch(self, client):
        payload = {"requests": [_online_request("TXN-P"), _fallback_request("TXN-F"), _phone_request("TXN-FULL")]}
        with patch("random.uniform", side_effect=_R):
            results = client.post("/api/v1/fusion/batch", json=payload).json()["results"]
        modes = {r["transaction_id"]: r["ensemble_mode"] for r in results}
        assert modes["TXN-P"]    == "partial"
        assert modes["TXN-F"]    == "fallback"
        assert modes["TXN-FULL"] == "full"

    def test_response_shape(self, client):
        payload = {"requests": [_online_request("TXN-B1")]}
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/batch", json=payload).json()
        for key in ("total_processed", "total_approved", "total_review", "total_blocked", "results"):
            assert key in body


class TestAuditEndpoints:

    def test_audit_list_empty_at_start(self, client):
        assert client.get("/api/v1/audit").json() == []

    def test_unknown_id_returns_404(self, client):
        assert client.get("/api/v1/audit/does-not-exist").status_code == 404

    def test_record_retrievable_after_evaluate(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        rid = body["audit"]["request_id"]
        r = client.get(f"/api/v1/audit/{rid}")
        assert r.status_code == 200
        assert r.json()["request_id"] == rid

    def test_record_decision_matches_response(self, client):
        with patch("random.uniform", side_effect=_R):
            eval_body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        record = client.get(f"/api/v1/audit/{eval_body['audit']['request_id']}").json()
        assert record["decision"] == eval_body["decision"]

    def test_record_fraud_score_matches_response(self, client):
        with patch("random.uniform", side_effect=_R):
            eval_body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        record = client.get(f"/api/v1/audit/{eval_body['audit']['request_id']}").json()
        assert abs(record["fraud_score"] - eval_body["fraud_score"]) < 1e-6

    def test_list_grows_with_requests(self, client):
        for i in range(3):
            with patch("random.uniform", side_effect=_R):
                client.post("/api/v1/fusion/evaluate", json=_online_request(f"TXN-{i}"))
        assert len(client.get("/api/v1/audit").json()) == 3

    def test_batch_creates_one_entry_per_transaction(self, client):
        payload = {"requests": [_online_request(f"TXN-{i}") for i in range(3)]}
        with patch("random.uniform", side_effect=_R):
            client.post("/api/v1/fusion/batch", json=payload)
        assert len(client.get("/api/v1/audit").json()) == 3

    def test_record_has_ensemble_mode(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_fallback_request()).json()
        record = client.get(f"/api/v1/audit/{body['audit']['request_id']}").json()
        assert record["ensemble_mode"] == "fallback"

    def test_record_has_model_version(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        record = client.get(f"/api/v1/audit/{body['audit']['request_id']}").json()
        assert "model_version" in record

    def test_record_has_timestamp(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        record = client.get(f"/api/v1/audit/{body['audit']['request_id']}").json()
        assert "T" in record["timestamp"]

    def test_record_has_modality_fields(self, client):
        with patch("random.uniform", side_effect=_R):
            body = client.post("/api/v1/fusion/evaluate", json=_online_request()).json()
        record = client.get(f"/api/v1/audit/{body['audit']['request_id']}").json()
        for field in ("modalities_requested", "modalities_used", "modalities_failed"):
            assert field in record