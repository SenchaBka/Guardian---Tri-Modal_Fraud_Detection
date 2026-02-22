"""guardian.nlp.preprocessor

Preprocessing + canonicalization for the NLP Stream.

Responsibilities of this module:
- Receive the *raw request* sent by the Orchestration/Fusion layer (RawNLPRequest)
- Validate + normalize text fields
- (Optional) Mask basic PII patterns
- Produce a stable, canonical NLPInput used by downstream modules (classifier/NER/etc.)

IMPORTANT:
- The Fusion layer is responsible for mapping the upstream (UI/demo) input into RawNLPRequest.
- This module is responsible for converting RawNLPRequest -> NLPInput.

This design keeps ownership boundaries clean and avoids duplicated logic across services.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from .api.schemas import NLPInput, RawNLPRequest, TextSource


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing configuration.

    Keep it simple for Iteration #1. If needed later, move to settings/env.
    """

    lowercase: bool = False  # keep False by default to preserve casing cues (e.g., brand names)
    collapse_whitespace: bool = True
    max_chars: int = 6000  # hard cap to protect the model/API
    mask_pii: bool = True


DEFAULT_CFG = PreprocessConfig()


# -----------------------------
# PII masking (basic / heuristic)
# -----------------------------


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def mask_basic_pii(text: str) -> str:
    """Mask common PII patterns.

    NOTE: This is *not* a perfect PII detector. It's a pragmatic MVP safeguard.
    """

    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    # Avoid masking amounts like 9839.64 (has dot). The card regex targets long digit runs.
    text = _CARD_RE.sub("[CARD]", text)
    return text


# -----------------------------
# Text cleaning
# -----------------------------


def clean_text(raw_text: str, cfg: PreprocessConfig = DEFAULT_CFG) -> str:
    """Normalize text for model consumption."""

    if raw_text is None:
        return ""

    text = str(raw_text)

    # Normalize newlines/tabs early
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    if cfg.collapse_whitespace:
        # Collapse multiple whitespace/newlines into single spaces
        text = re.sub(r"\s+", " ", text).strip()
    else:
        text = text.strip()

    if cfg.lowercase:
        text = text.lower()

    if cfg.mask_pii:
        text = mask_basic_pii(text)

    # Truncate to protect the service/model
    if cfg.max_chars and len(text) > cfg.max_chars:
        text = text[: cfg.max_chars].rstrip() + " …"

    return text


# -----------------------------
# Canonicalization (RawNLPRequest -> NLPInput)
# -----------------------------


def _choose_text_source(raw: RawNLPRequest) -> Tuple[TextSource, str]:
    """Pick the best available text source.

    Priority order reflects typical transaction pipelines:
    invoice > merchant > narrative > ticket

    If multiple sources exist, we still may combine them elsewhere.
    """

    p = raw.payload

    candidates = [
        ("invoice", p.invoice_text),
        ("merchant", p.merchant_text),
        ("narrative", p.narrative_text),
        ("ticket", p.ticket_text),
    ]

    for src, txt in candidates:
        if txt is not None and str(txt).strip() != "":
            return src, str(txt)

    # RawNLPRequest validation should prevent reaching here
    return "combined", ""  # type: ignore


def _combine_texts(raw: RawNLPRequest) -> Tuple[TextSource, str]:
    """Combine all available text fields into one canonical text.

    We add short headers to preserve provenance while still producing a single string.
    """

    p = raw.payload

    parts = []
    if p.merchant_text and str(p.merchant_text).strip():
        parts.append(f"[MERCHANT] {p.merchant_text}")
    if p.narrative_text and str(p.narrative_text).strip():
        parts.append(f"[NARRATIVE] {p.narrative_text}")
    if p.invoice_text and str(p.invoice_text).strip():
        parts.append(f"[INVOICE] {p.invoice_text}")
    if p.ticket_text and str(p.ticket_text).strip():
        parts.append(f"[TICKET] {p.ticket_text}")

    if not parts:
        # RawNLPRequest validation should prevent reaching here
        return "combined", ""  # type: ignore

    if len(parts) == 1:
        # Preserve original source if only one
        src, txt = _choose_text_source(raw)
        return src, txt

    return "combined", "\n".join(parts)


def raw_request_to_nlp_input(
    raw: RawNLPRequest,
    cfg: PreprocessConfig = DEFAULT_CFG,
    combine_sources: bool = True,
) -> NLPInput:
    """Convert a RawNLPRequest to the canonical NLPInput.

    Args:
        raw: Validated RawNLPRequest (Fusion -> NLP)
        cfg: Preprocessing config
        combine_sources: If True, concatenates all available text fields.

    Returns:
        NLPInput ready for downstream NLP modules.
    """

    if combine_sources:
        text_source, text = _combine_texts(raw)
    else:
        text_source, text = _choose_text_source(raw)

    cleaned = clean_text(text, cfg=cfg)

    # If after cleaning we lose all text, keep a minimal placeholder.
    # Downstream modules can treat it as degraded.
    if cleaned.strip() == "":
        cleaned = "[NO_TEXT]"
        text_source = "combined"

    return NLPInput(
        transaction_id=raw.transaction_id,
        text=cleaned,
        text_source=text_source,
        language=raw.language or "en",
        metadata=raw.metadata,
    )


def load_jsonl_as_raw_requests(jsonl_path: str) -> list[RawNLPRequest]:
    """Helper for local testing with your interim JSONL samples.

    Expected JSONL lines match the RawNLPRequest schema.
    This is optional (dev utility) and not used by the API directly.
    """

    import json

    out: list[RawNLPRequest] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(RawNLPRequest(**obj))
    return out