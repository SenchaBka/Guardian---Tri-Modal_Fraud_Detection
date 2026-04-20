"""
conftest.py – Shared fixtures for Guardian Fusion Layer tests.

Key design decisions
--------------------
* random.uniform is patched to a deterministic side_effect list in every
  test that calls orchestrator mocks.  Without this, _mock_numerical /
  _mock_nlp / _mock_voice produce different scores on every run, making
  assertions impossible.
* The audit store is cleared before and after every test via autouse.
* A reusable minimal FusionRequest fixture covers the common case.
"""

import pytest
from datetime import datetime
from fusion.schemas import (
    FusionRequest, TransactionData, TextPayload, VoicePayload,
    AvailableModalities, FusionOptions, ChannelType,
)
from fusion import audit as audit_store


# ---------------------------------------------------------------------------
# Audit isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_audit():
    audit_store.clear()
    yield
    audit_store.clear()


# ---------------------------------------------------------------------------
# Minimal request builders
# ---------------------------------------------------------------------------

def make_request(
    txn_id="TXN-001",
    amount=250.0,
    channel=ChannelType.ONLINE,
    country="US",
    merchant_category="retail",
    nlp=True,
    voice=False,
    merchant_text="Amazon purchase",
    narrative_text="online shopping",
    audio_url=None,
    explain=True,
    threshold_override=None,
    force_review_on_fallback=True,
) -> FusionRequest:
    text_payload  = TextPayload(merchant_text=merchant_text, narrative_text=narrative_text) if nlp else None
    voice_payload = VoicePayload(audio_url=audio_url or "http://example.com/audio.wav") if voice else None
    return FusionRequest(
        transaction_id=txn_id,
        available_modalities=AvailableModalities(nlp=nlp, voice=voice),
        transaction_data=TransactionData(
            amount=amount,
            currency="USD",
            timestamp=datetime(2024, 6, 1, 12, 0),
            channel=channel,
            country=country,
            merchant_category=merchant_category,
        ),
        text_payload=text_payload,
        voice_payload=voice_payload,
        options=FusionOptions(
            explain=explain,
            threshold_override=threshold_override,
            force_review_on_fallback=force_review_on_fallback,
        ),
    )


@pytest.fixture
def basic_request():
    return make_request()


@pytest.fixture
def fallback_request():
    return make_request(nlp=False, voice=False)


@pytest.fixture
def full_request():
    return make_request(nlp=True, voice=True, audio_url="http://example.com/audio.wav")


@pytest.fixture
def high_amount_request():
    return make_request(amount=50_000.0, country="NG", merchant_category="crypto")