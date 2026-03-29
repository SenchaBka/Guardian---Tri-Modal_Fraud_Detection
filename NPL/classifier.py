"""guardian.nlp.classifier

NLP Stream classifier utilities.

Iteration 2 goal:
- Use the locally fine-tuned PaySim FinBERT model to produce a real fraud score.
- Apply the calibrated validation threshold selected during training.
- Keep a heuristic fallback only if the local model cannot load.

Important:
- We no longer use sentiment logits as a risk proxy.
- The transformer now performs direct binary fraud classification.
- `score` represents the model probability for the fraud class.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

from .model_loader import load_trained_nlp_bundle

logger = logging.getLogger(__name__)


# -----------------------------
# Utilities
# -----------------------------


def _softmax(logits) -> list[float]:
    """Numerically-stable softmax for a 1D list/array."""

    m = max(float(x) for x in logits)
    exps = [math.exp(float(x) - m) for x in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# -----------------------------
# Heuristic fallback
# -----------------------------


_FRAUD_CUES = [
    # urgency / coercion
    "urgent",
    "immediately",
    "asap",
    "right away",
    "wire",
    "transfer now",
    "send now",
    "do not delay",
    # disputes / chargebacks
    "chargeback",
    "dispute",
    "fraudulent",
    "unauthorized",
    "scam",
    "phishing",
    "account compromised",
    # risky terms
    "gift card",
    "crypto",
    "bitcoin",
    "refund",
    "overpayment",
    "invoice",
]


def heuristic_semantic_risk(text: str) -> float:
    """A tiny keyword-based fallback risk score in [0, 1].

    This is ONLY used when the transformer model cannot run.
    """

    t = (text or "").lower()
    if not t.strip():
        return 0.5

    hits = 0
    for cue in _FRAUD_CUES:
        if cue in t:
            hits += 1

    # Add a small bump if there are many exclamation marks or ALL CAPS words
    exclam = min(t.count("!"), 5)
    caps_words = len(re.findall(r"\b[A-Z]{4,}\b", text or ""))

    score = 0.12 * hits + 0.03 * exclam + 0.02 * min(caps_words, 5)
    return _clip01(score)


# -----------------------------
# FinBERT-based semantic risk
# -----------------------------


@dataclass(frozen=True)
class SemanticRiskResult:
    score: float
    model_version: str
    details: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None


def get_semantic_risk(text: str, *, max_length: int = 256) -> SemanticRiskResult:
    """Compute fraud probability score in [0, 1] for an input text.

    Args:
        text: Canonicalized text from NLPInput
        max_length: Token cap for transformer input

    Returns:
        SemanticRiskResult(score, model_version, details, threshold)

    Behavior:
    - If the trained local fraud model is available, returns the fraud probability.
    - Otherwise returns a heuristic fallback risk.
    """

    cleaned = (text or "").strip()

    try:
        logger.info("Running NLP inference...")
        bundle = load_trained_nlp_bundle()
        tokenizer = bundle.tokenizer
        model = bundle.model
        device = bundle.device

        import torch

        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # Move inputs to device
        if device in {"cuda", "mps"}:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(0).detach().cpu().tolist()

        probs = _softmax(logits)

        # Binary fraud classifier convention used in Iteration 2:
        # class 0 = non-fraud, class 1 = fraud
        fraud_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

        id2label = getattr(model.config, "id2label", {}) or {}
        label_probs: Dict[str, float] = {}
        for i, p in enumerate(probs):
            raw_name = str(id2label.get(i, i)).lower()
            if raw_name in {"0", "label_0", "non_fraud", "non-fraud", "legit", "negative"}:
                name = "non_fraud"
            elif raw_name in {"1", "label_1", "fraud", "positive"}:
                name = "fraud"
            else:
                name = raw_name
            label_probs[name] = float(p)

        label_probs["fraud_probability"] = fraud_prob
        label_probs["threshold"] = float(bundle.threshold)
        label_probs["predicted_fraud"] = float(fraud_prob >= bundle.threshold)
        logger.info("Fraud probability: %.6f", fraud_prob)
        if abs(fraud_prob - float(bundle.threshold)) < 0.05:
            logger.warning("Very low confidence prediction")

        return SemanticRiskResult(
            score=_clip01(fraud_prob),
            model_version=bundle.model_name,
            details=label_probs,
            threshold=float(bundle.threshold),
        )

    except Exception:
        # Keep API functional even if model is unavailable
        logger.error("NLP inference failed, using heuristic fallback", exc_info=True)
        fallback_score = heuristic_semantic_risk(cleaned)
        return SemanticRiskResult(
            score=fallback_score,
            model_version="heuristic_v1",
            details={
                "fraud_probability": fallback_score,
                "fallback_used": 1.0,
            },
            threshold=None,
        )
