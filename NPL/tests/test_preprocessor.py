import json
import tempfile
import unittest
from pathlib import Path

from NPL.api.schemas import NLPMetadata, NLPTextPayload, RawNLPRequest
from NPL.preprocessor import (
    PreprocessConfig,
    _choose_text_source,
    clean_text,
    load_jsonl_as_raw_requests,
    mask_basic_pii,
    raw_request_to_nlp_input,
)


def build_raw_request(**payload_overrides):
    payload_data = {
        "merchant_text": None,
        "narrative_text": None,
        "invoice_text": None,
        "ticket_text": None,
    }
    payload_data.update(payload_overrides)
    payload = NLPTextPayload(**payload_data)
    return RawNLPRequest(
        transaction_id="txn-123",
        language="en",
        payload=payload,
        metadata=NLPMetadata(amount=10.5, currency="CAD"),
    )


class MaskBasicPIITests(unittest.TestCase):
    def test_masks_email_phone_and_card(self):
        text = "Contact fraud@bank.com or 416-555-0100 with 4111 1111 1111 1111."

        masked = mask_basic_pii(text)

        self.assertIn("[EMAIL]", masked)
        self.assertIn("[PHONE]", masked)
        self.assertIn("[CARD]", masked)


class CleanTextTests(unittest.TestCase):
    def test_returns_empty_string_for_none(self):
        self.assertEqual(clean_text(None), "")

    def test_normalizes_whitespace_and_masks_pii(self):
        text = "  Call\tme at 416-555-0100 \r\n ASAP  "

        cleaned = clean_text(text)

        self.assertEqual(cleaned, "Call me at [PHONE] ASAP")

    def test_respects_lowercase_and_no_collapse(self):
        cfg = PreprocessConfig(lowercase=True, collapse_whitespace=False, mask_pii=False)

        cleaned = clean_text("  Hello\tWorld  ", cfg=cfg)

        self.assertEqual(cleaned, "hello world")

    def test_truncates_long_text_with_ellipsis(self):
        cfg = PreprocessConfig(max_chars=5, mask_pii=False)

        cleaned = clean_text("abcdefghi", cfg=cfg)

        self.assertEqual(cleaned, "abcde …")


class RawRequestToNLPInputTests(unittest.TestCase):
    def test_choose_text_source_prefers_invoice(self):
        raw = build_raw_request(
            merchant_text="merchant copy",
            invoice_text="invoice copy",
            narrative_text="narrative copy",
        )

        source, text = _choose_text_source(raw)

        self.assertEqual(source, "invoice")
        self.assertEqual(text, "invoice copy")

    def test_combines_sources_when_requested(self):
        raw = build_raw_request(
            merchant_text="ACME",
            narrative_text="urgent refund",
        )

        result = raw_request_to_nlp_input(raw, combine_sources=True)

        self.assertEqual(result.text_source, "combined")
        self.assertIn("[MERCHANT] ACME", result.text)
        self.assertIn("[NARRATIVE] urgent refund", result.text)

    def test_preserves_single_source_when_not_combining(self):
        raw = build_raw_request(merchant_text="Single source")

        result = raw_request_to_nlp_input(raw, combine_sources=False)

        self.assertEqual(result.text_source, "merchant")
        self.assertEqual(result.text, "Single source")

    def test_uses_placeholder_when_cleaning_removes_all_text(self):
        raw = build_raw_request(merchant_text="placeholder")
        raw.payload.merchant_text = "   "

        result = raw_request_to_nlp_input(
            raw,
            cfg=PreprocessConfig(mask_pii=False),
            combine_sources=False,
        )

        self.assertEqual(result.text, "[NO_TEXT]")
        self.assertEqual(result.text_source, "combined")


class LoadJsonlAsRawRequestsTests(unittest.TestCase):
    def test_loads_requests_and_skips_blank_lines(self):
        request_1 = {
            "transaction_id": "txn-1",
            "language": "en",
            "payload": {"merchant_text": "hello"},
            "metadata": {"amount": 10},
        }
        request_2 = {
            "transaction_id": "txn-2",
            "language": "fr",
            "payload": {"invoice_text": "bonjour"},
            "metadata": {"amount": 20},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            path.write_text(
                "\n".join([json.dumps(request_1), "", json.dumps(request_2)]),
                encoding="utf-8",
            )

            requests = load_jsonl_as_raw_requests(str(path))

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].transaction_id, "txn-1")
        self.assertEqual(requests[1].payload.invoice_text, "bonjour")
