import unittest
from unittest import mock

from fastapi import HTTPException

from NPL.api.api import health, score_nlp
from NPL.api.schemas import NLPStatus, NLPTextPayload, RawNLPRequest
from NPL.classifier import SemanticRiskResult


def build_request(text="merchant details"):
    return RawNLPRequest(
        transaction_id="txn-1",
        language="en",
        payload=NLPTextPayload(merchant_text=text),
    )


class HealthEndpointTests(unittest.TestCase):
    def test_health_includes_loaded_bundle_details(self):
        bundle = mock.Mock(
            threshold=0.3,
            model_name="finbert-local",
            device="cpu",
            source="huggingface",
            revision="main",
            backend="transformer",
        )

        with mock.patch("NPL.api.api.healthcheck_model", return_value=(True, "loaded")):
            with mock.patch("NPL.api.api.load_trained_nlp_bundle", return_value=bundle):
                response = health()

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["threshold"], 0.3)
        self.assertEqual(response["model_version"], "finbert-local")
        self.assertEqual(response["device"], "cpu")
        self.assertEqual(response["model_backend"], "transformer")

    def test_health_degrades_when_model_load_fails(self):
        with mock.patch(
            "NPL.api.api.healthcheck_model",
            return_value=(False, "model load failed"),
        ):
            with mock.patch(
                "NPL.api.api.load_trained_nlp_bundle",
                side_effect=RuntimeError("boom"),
            ):
                response = health()

        self.assertEqual(response, {"status": "degraded", "model": "model load failed"})


class ScoreNLPTests(unittest.TestCase):
    def test_score_nlp_returns_ok_status_for_model_prediction(self):
        sem = SemanticRiskResult(
            score=0.91,
            model_version="finbert-local",
            details={"predicted_fraud": 1.0},
            threshold=0.5,
        )

        with mock.patch("NPL.api.api.get_semantic_risk", return_value=sem):
            result = score_nlp(build_request("Detailed merchant narrative"))

        self.assertEqual(result.status, NLPStatus.ok)
        self.assertEqual(result.score_nlp, 0.91)
        self.assertEqual(result.signals.semantic_risk, 0.91)

    def test_score_nlp_marks_short_text_as_degraded(self):
        sem = SemanticRiskResult(
            score=0.22,
            model_version="finbert-local",
            details={"predicted_fraud": 0.0},
            threshold=0.5,
        )

        with mock.patch("NPL.api.api.get_semantic_risk", return_value=sem):
            result = score_nlp(build_request("short"))

        self.assertEqual(result.status, NLPStatus.degraded)

    def test_score_nlp_marks_heuristic_fallback_as_degraded(self):
        sem = SemanticRiskResult(
            score=0.4,
            model_version="heuristic_v1",
            details={"fallback_used": 1.0},
            threshold=None,
        )

        with mock.patch("NPL.api.api.get_semantic_risk", return_value=sem):
            result = score_nlp(build_request("normal merchant content"))

        self.assertEqual(result.status, NLPStatus.degraded)
        self.assertIsNone(result.signals.typosquatting_risk)

    def test_score_nlp_wraps_preprocessing_failures_as_http_422(self):
        req = build_request("merchant details")

        with mock.patch(
            "NPL.api.api.raw_request_to_nlp_input",
            side_effect=ValueError("invalid payload"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                score_nlp(req)

        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.detail, "invalid payload")
