"""guardian.nlp.model_loader

Model loading utilities for the NLP Stream.

Current goal:
- Load the trained fraud classifier ONCE (lazy) and reuse it across requests.
- Prefer the Hugging Face hosted model for Iteration 2 serving.
- Expose the tokenizer, model, device, source, revision, and calibrated fraud threshold.

Notes:
- Iteration 2 can now load the promoted model directly from Hugging Face.
- A local checkpoint fallback is still supported if the hosted model cannot be reached.
- This keeps the API stable while allowing future model promotion without changing serving code.

Environment variables (optional):
- NLP_MODEL_NAME: Hugging Face model repo id
  (default: Lmateosl/guardian-finbert-npl)
- NLP_MODEL_REVISION: Hugging Face revision or commit hash
  (default: main)
- NLP_MODEL_DIR: local fallback directory of the trained checkpoint
  (default: models/nlp/finbert/paysim_sample100k_ep2)
- NLP_THRESHOLD: calibrated fraud threshold (default: 0.0039)
- NLP_DEVICE: 'cpu', 'cuda', or 'mps' (default: auto)
- NLP_CACHE_DIR: optional custom Hugging Face cache directory
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    """Container for the NLP model artifacts."""

    model_name: str
    model: object
    device: str
    source: str
    revision: str
    threshold: float
    tokenizer: object | None = None
    vectorizer: object | None = None
    backend: str = "transformer"


def _resolve_device() -> str:
    """Pick a device.

    - If NLP_DEVICE is set, respect it.
    - Otherwise use CUDA if available, else MPS if available, else CPU.
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

    model_kind = os.getenv("NLP_MODEL_KIND", "transformer").strip().lower()
    default_model_name = (
        "Lmateosl/guardian-linear-svm-npl"
        if model_kind in {"sklearn", "linear_svm", "baseline"}
        else "Lmateosl/guardian-finbert-npl"
    )
    default_model_dir = (
        "models/nlp/baseline/combined_200k_linear_svm"
        if model_kind in {"sklearn", "linear_svm", "baseline"}
        else "models/nlp/finbert/paysim_sample100k_ep2"
    )

    model_name = os.getenv("NLP_MODEL_NAME", default_model_name)
    revision = os.getenv("NLP_MODEL_REVISION", "main")
    model_dir = os.getenv("NLP_MODEL_DIR", default_model_dir)
    threshold_override = os.getenv("NLP_THRESHOLD", "").strip()
    cache_dir = os.getenv("NLP_CACHE_DIR", "").strip() or None
    device = _resolve_device()
    logger.info("Loading NLP model...")
    logger.info("Configured NLP backend: %s", model_kind)
    logger.info("Preferred Hugging Face model: %s @ %s", model_name, revision)
    logger.info("Local fallback path: %s", model_dir)

    if model_kind in {"sklearn", "linear_svm", "baseline"}:
        return _load_sklearn_bundle(
            model_name=model_name,
            revision=revision,
            model_dir=model_dir,
            cache_dir=cache_dir,
            threshold_override=threshold_override,
        )

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

    threshold = float(threshold_override or "0.0039")
    source = "huggingface"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        )
        logger.info("Loaded NLP model from Hugging Face")
    except Exception:
        logger.warning(
            "Failed to load model from Hugging Face; falling back to local checkpoint",
            exc_info=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        source = "local"
        revision = "local"
        model_name = model_dir
        logger.info("Loaded NLP model from local checkpoint")

    try:
        if device in {"cuda", "mps"}:
            model.to(device)
        model.eval()
    except Exception:
        logger.error("Failed to finalize model on device", exc_info=True)
        raise

    logger.info("Model loaded successfully")

    return ModelBundle(
        model_name=model_name,
        model=model,
        device=device,
        source=source,
        revision=revision,
        threshold=threshold,
        tokenizer=tokenizer,
        backend="transformer",
    )


def _resolve_threshold_from_metadata(bundle_dir: Path, threshold_override: str) -> float:
    if threshold_override:
        return float(threshold_override)

    metadata_path = bundle_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "threshold" in metadata:
            return float(metadata["threshold"])

    return 0.5


def _load_sklearn_bundle(
    *,
    model_name: str,
    revision: str,
    model_dir: str,
    cache_dir: str | None,
    threshold_override: str,
) -> ModelBundle:
    model_file = os.getenv("NLP_SKLEARN_MODEL_FILE", "linear_svm_model.joblib")
    vectorizer_file = os.getenv("NLP_SKLEARN_VECTORIZER_FILE", "tfidf_vectorizer.joblib")
    source = "huggingface"
    bundle_dir: Path | None = None

    try:
        from huggingface_hub import snapshot_download
        import joblib

        snapshot_path = snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=cache_dir,
        )
        bundle_dir = Path(snapshot_path)
        logger.info("Loaded sklearn NLP bundle from Hugging Face")
    except Exception:
        logger.warning(
            "Failed to load sklearn bundle from Hugging Face; falling back to local directory",
            exc_info=True,
        )
        import joblib

        bundle_dir = Path(model_dir)
        source = "local"
        revision = "local"
        model_name = model_dir

    model_path = bundle_dir / model_file
    vectorizer_path = bundle_dir / vectorizer_file
    if not model_path.exists():
        raise RuntimeError(f"Missing sklearn model artifact: {model_path}")
    if not vectorizer_path.exists():
        raise RuntimeError(f"Missing sklearn vectorizer artifact: {vectorizer_path}")

    import joblib

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    threshold = _resolve_threshold_from_metadata(bundle_dir, threshold_override)
    logger.info("Sklearn NLP bundle loaded successfully")

    return ModelBundle(
        model_name=model_name,
        model=model,
        vectorizer=vectorizer,
        device="cpu",
        source=source,
        revision=revision,
        threshold=threshold,
        backend="sklearn",
    )


def healthcheck_model() -> Tuple[bool, str]:
    """Best-effort healthcheck.

    Used by api.py to confirm the model can load.
    """

    try:
        b = load_trained_nlp_bundle()
        return (
            True,
            f"loaded {b.model_name} ({b.source}@{b.revision}) on {b.device} | threshold={b.threshold}",
        )
    except Exception as e:
        logger.error("Failed to load model", exc_info=True)
        return False, f"model load failed: {e}"


# Backwards-compatible alias used by earlier Iteration 1 code.
load_finbert_bundle = load_trained_nlp_bundle
