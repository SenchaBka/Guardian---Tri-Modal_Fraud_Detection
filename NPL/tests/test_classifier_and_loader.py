import os
import types
import unittest
from unittest import mock

from NPL.classifier import (
    _clip01,
    _softmax,
    get_semantic_risk,
    heuristic_semantic_risk,
)
from NPL.model_loader import ModelBundle, healthcheck_model, load_trained_nlp_bundle


class UtilityTests(unittest.TestCase):
    def test_softmax_handles_large_logits(self):
        probs = _softmax([1000, 1001])

        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertGreater(probs[1], probs[0])

    def test_clip01_limits_negative_and_large_values(self):
        self.assertEqual(_clip01(-5), 0.0)
        self.assertEqual(_clip01(3), 1.0)
        self.assertEqual(_clip01(0.4), 0.4)

    def test_heuristic_risk_returns_midpoint_for_empty_text(self):
        self.assertEqual(heuristic_semantic_risk(""), 0.5)

    def test_heuristic_risk_detects_multiple_cues_and_caps(self):
        score = heuristic_semantic_risk("URGENT refund!!! send now")

        self.assertGreater(score, 0.2)
        self.assertLessEqual(score, 1.0)


class GetSemanticRiskTests(unittest.TestCase):
    def test_returns_model_probability_when_bundle_loads(self):
        fake_inputs = {"input_ids": types.SimpleNamespace(to=lambda *_: "moved")}
        fake_tokenizer = mock.Mock(return_value=fake_inputs)
        fake_model = mock.Mock()
        fake_model.config = types.SimpleNamespace(id2label={0: "non_fraud", 1: "fraud"})
        fake_model.return_value = types.SimpleNamespace(
            logits=types.SimpleNamespace(
                squeeze=lambda *_: types.SimpleNamespace(
                    detach=lambda: types.SimpleNamespace(
                        cpu=lambda: types.SimpleNamespace(tolist=lambda: [0.1, 2.1])
                    )
                )
            )
        )
        bundle = ModelBundle(
            model_name="finbert-local",
            tokenizer=fake_tokenizer,
            model=fake_model,
            device="cpu",
            threshold=0.7,
        )
        fake_torch = types.SimpleNamespace(
            no_grad=lambda: mock.MagicMock(__enter__=lambda *_: None, __exit__=lambda *_: False)
        )

        with mock.patch("NPL.classifier.load_trained_nlp_bundle", return_value=bundle):
            with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                result = get_semantic_risk("wire transfer now")

        self.assertEqual(result.model_version, "finbert-local")
        self.assertAlmostEqual(result.threshold, 0.7)
        self.assertGreater(result.score, 0.8)
        self.assertEqual(result.details["predicted_fraud"], 1.0)

    def test_falls_back_to_heuristic_when_loading_fails(self):
        with mock.patch(
            "NPL.classifier.load_trained_nlp_bundle",
            side_effect=RuntimeError("missing model"),
        ):
            result = get_semantic_risk("urgent refund")

        self.assertEqual(result.model_version, "heuristic_v1")
        self.assertIsNone(result.threshold)
        self.assertEqual(result.details["fallback_used"], 1.0)


class ResolveDeviceTests(unittest.TestCase):
    def tearDown(self):
        load_trained_nlp_bundle.cache_clear()

    def test_load_bundle_uses_env_overrides_and_moves_model_to_device(self):
        fake_tokenizer = mock.Mock(name="tokenizer")
        fake_model = mock.Mock(name="model")
        moved_model = mock.Mock(name="moved_model")
        fake_model.to.return_value = moved_model
        fake_auto_tokenizer = mock.Mock(from_pretrained=mock.Mock(return_value=fake_tokenizer))
        fake_auto_model = mock.Mock(from_pretrained=mock.Mock(return_value=fake_model))
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=fake_auto_tokenizer,
            AutoModelForSequenceClassification=fake_auto_model,
        )
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        )

        with mock.patch.dict(
            os.environ,
            {
                "NLP_MODEL_DIR": "custom-model",
                "NLP_THRESHOLD": "0.2",
                "NLP_DEVICE": "cuda",
            },
            clear=False,
        ):
            with mock.patch.dict(
                "sys.modules",
                {
                    "transformers": fake_transformers,
                    "torch": fake_torch,
                },
            ):
                bundle = load_trained_nlp_bundle()

        self.assertEqual(bundle.model_name, "custom-model")
        self.assertEqual(bundle.threshold, 0.2)
        self.assertEqual(bundle.device, "cuda")
        fake_auto_tokenizer.from_pretrained.assert_called_once_with("custom-model")
        fake_auto_model.from_pretrained.assert_called_once_with("custom-model")
        fake_model.to.assert_called_once_with("cuda")
        moved_model.eval.assert_called_once()

    def test_healthcheck_model_returns_failure_message(self):
        with mock.patch(
            "NPL.model_loader.load_trained_nlp_bundle",
            side_effect=RuntimeError("boom"),
        ):
            ok, msg = healthcheck_model()

        self.assertFalse(ok)
        self.assertIn("model load failed: boom", msg)
