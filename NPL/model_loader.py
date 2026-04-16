"""guardian.nlp.model_loader

Model loading utilities for the NLP Stream.

Current goal:
- Load the trained PaySim FinBERT model ONCE (lazy) and reuse it across requests.
- Expose the tokenizer, model, device, and calibrated fraud threshold.

Notes:
- Iteration 2 now uses the fine-tuned local checkpoint instead of the generic
  FinBERT sentiment model from HuggingFace.
- If needed, the model path and threshold can still be overridden through
  environment variables.

Environment variables (optional):
- NLP_MODEL_DIR: local directory of the trained checkpoint
  (default: models/nlp/finbert/paysim_sample100k_ep2)
- NLP_THRESHOLD: calibrated fraud threshold (default: 0.0039)
- NLP_DEVICE: 'cpu', 'cuda', or 'mps' (default: auto)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    """Container for the NLP model artifacts."""

    model_name: str
    tokenizer: object
    model: object
    device: str
    threshold: float


def _resolve_device() -> str:
    """Pick a device.

    - If NLP_DEVICE is set, respect it.
    - Otherwise use CUDA if available, else CPU.
    """

    forced = os.getenv("NLP_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda", "mps"}:
        return forced

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # On Apple Silicon you may want 'mps' if available
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


@lru_cache(maxsize=1)
def load_trained_nlp_bundle() -> ModelBundle:
    """Lazy-load and cache the trained NLP model bundle.

    Returns:
        ModelBundle(tokenizer, model, device, threshold)

    Raises:
        RuntimeError if transformers/torch are not installed.
    """

    model_dir = os.getenv(
        "NLP_MODEL_DIR", "models/nlp/finbert/paysim_sample100k_ep2"
    )
    threshold = float(os.getenv("NLP_THRESHOLD", "0.0039"))
    device = _resolve_device()
    logger.info("Loading FinBERT model...")
    logger.info("Model path: %s", model_dir)

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        logger.error("Failed to load model", exc_info=True)
        raise RuntimeError(
            "transformers is required. Install with: pip install transformers"
        ) from e

    try:
        import torch
    except Exception as e:
        logger.error("Failed to load model", exc_info=True)
        raise RuntimeError("torch is required. Install with: pip install torch") from e

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        if device in {"cuda", "mps"}:
            model = model.to(device)

        model.eval()
    except Exception:
        logger.error("Failed to load model", exc_info=True)
        raise

    logger.info("Model loaded successfully")

    return ModelBundle(
        model_name=model_dir,
        tokenizer=tokenizer,
        model=model,
        device=device,
        threshold=threshold,
    )


def healthcheck_model() -> Tuple[bool, str]:
    """Best-effort healthcheck.

    Used by api.py to confirm the model can load.
    """

    try:
        b = load_trained_nlp_bundle()
        return True, f"loaded {b.model_name} on {b.device} | threshold={b.threshold}"
    except Exception as e:
        logger.error("Failed to load model", exc_info=True)
        return False, f"model load failed: {e}"


# Backwards-compatible alias used by earlier Iteration 1 code.
load_finbert_bundle = load_trained_nlp_bundle
