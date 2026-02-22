"""guardian.nlp.model_loader

Model loading utilities for the NLP Stream (Iteration #1 MVP).

Goal:
- Load the transformer model ONCE (lazy) and reuse it across requests.
- Provide a simple interface for downstream modules (classifier.py).

Notes:
- For the MVP we support FinBERT-style models from HuggingFace.
- If the model isn't available locally, HF will try to download it.
  (For demo environments without internet, pre-download or vendor the model.)

Environment variables (optional):
- NLP_MODEL_NAME: HuggingFace model id (default: ProsusAI/finbert)
- NLP_DEVICE: 'cpu' or 'cuda' (default: auto)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelBundle:
    """Container for the NLP model artifacts."""

    model_name: str
    tokenizer: object
    model: object
    device: str


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
def load_finbert_bundle() -> ModelBundle:
    """Lazy-load and cache the FinBERT bundle.

    Returns:
        ModelBundle(tokenizer, model, device)

    Raises:
        RuntimeError if transformers/torch are not installed.
    """

    model_name = os.getenv("NLP_MODEL_NAME", "ProsusAI/finbert")
    device = _resolve_device()

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "transformers is required. Install with: pip install transformers"
        ) from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required. Install with: pip install torch") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Move model to device
    if device in {"cuda", "mps"}:
        model = model.to(device)

    model.eval()

    return ModelBundle(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def healthcheck_model() -> Tuple[bool, str]:
    """Best-effort healthcheck.

    Used by api.py to confirm the model can load.
    """

    try:
        b = load_finbert_bundle()
        return True, f"loaded {b.model_name} on {b.device}"
    except Exception as e:
        return False, f"model load failed: {e}"