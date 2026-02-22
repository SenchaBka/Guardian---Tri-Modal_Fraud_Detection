"""guardian.nlp.classifier

NLP Stream classifier utilities.

Iteration #1 MVP goal:
- Produce a normalized semantic risk score in [0, 1] for a given text.

Important note about FinBERT:
- Many commonly used "FinBERT" checkpoints (e.g., ProsusAI/finbert) are trained for
  financial sentiment classification (positive/negative/neutral), not fraud directly.
- For this capstone MVP, we convert sentiment logits into a risk proxy:
    semantic_risk := P(negative)
  This is a reasonable stand-in when the text resembles complaints, disputes, urgency,
  or anomalous narratives. Later iterations can replace this with a fraud-specific model.

If the transformer model cannot load (offline/no deps), we fall back to a lightweight
heuristic scorer so the API remains functional.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

from .model_loader import load_finbert_bundle


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


def get_semantic_risk(text: str, *, max_length: int = 256) -> SemanticRiskResult:
    """Compute semantic risk score in [0, 1] for an input text.

    Args:
        text: Canonicalized text from NLPInput
        max_length: Token cap for transformer input

    Returns:
        SemanticRiskResult(score, model_version, details)

    Behavior:
    - If FinBERT is available, returns P(negative) as risk proxy.
    - Otherwise returns heuristic risk.
    """

    cleaned = (text or "").strip()

    try:
        bundle = load_finbert_bundle()
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

        # Try to map label names if present
        id2label = getattr(model.config, "id2label", {}) or {}
        label_probs: Dict[str, float] = {}
        for i, p in enumerate(probs):
            name = str(id2label.get(i, i)).lower()
            label_probs[name] = float(p)

        # Heuristic mapping to "negative" class
        neg_keys = [k for k in label_probs.keys() if "neg" in k]
        if neg_keys:
            risk = label_probs[neg_keys[0]]
        else:
            # Common ordering for finbert sentiment is: [negative, neutral, positive]
            risk = probs[0] if len(probs) >= 1 else 0.5

        return SemanticRiskResult(
            score=_clip01(float(risk)),
            model_version=bundle.model_name,
            details=label_probs,
        )

    except Exception:
        # Keep API functional even if model is unavailable
        return SemanticRiskResult(
            score=heuristic_semantic_risk(cleaned),
            model_version="heuristic_v1",
            details=None,
        )