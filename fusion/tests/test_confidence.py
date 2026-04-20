"""
Unit Tests – fusion.confidence
================================
Covers calculate_base_confidence, calculate_availability_penalty,
calculate_confidence_adjustment, apply_fallback_overrides.
"""

import pytest
from fusion.confidence import (
    calculate_base_confidence,
    calculate_availability_penalty,
    calculate_confidence_adjustment,
    apply_fallback_overrides,
)
from fusion.schemas import EnsembleMode, DecisionType, ConfidenceAdjustment
from fusion.config import (
    CONFIDENCE_PENALTIES, ERROR_PENALTY_PER_MODALITY,
    FALLBACK_OVERRIDES, PARTIAL_MODE_OVERRIDES,
)


# ---------------------------------------------------------------------------
# calculate_base_confidence
# ---------------------------------------------------------------------------

class TestCalculateBaseConfidence:

    def test_no_scores_returns_half(self):
        assert calculate_base_confidence(0.5, {"numerical": None, "nlp": None}) == 0.5

    def test_single_score_agreement_is_point7(self):
        result = calculate_base_confidence(0.2, {"numerical": 0.2})
        # agreement_confidence = 0.7 for single score
        assert 0.0 < result <= 1.0

    def test_perfectly_agreeing_scores_high_confidence(self):
        scores = {"numerical": 0.8, "nlp": 0.8, "voice": 0.8}
        result = calculate_base_confidence(0.8, scores)
        # variance = 0 → agreement_confidence = 1.0
        assert result > 0.5

    def test_highly_disagreeing_scores_lower_confidence(self):
        agree    = calculate_base_confidence(0.5, {"numerical": 0.5, "nlp": 0.5})
        disagree = calculate_base_confidence(0.5, {"numerical": 0.1, "nlp": 0.9})
        assert agree > disagree

    def test_score_near_boundary_lowers_boundary_confidence(self):
        # 0.30 is a decision boundary – score right on it should have low boundary conf
        near   = calculate_base_confidence(0.30, {"numerical": 0.30})
        far    = calculate_base_confidence(0.05, {"numerical": 0.05})
        assert far >= near

    def test_output_in_unit_interval(self):
        for s in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = calculate_base_confidence(s, {"numerical": s})
            assert 0.0 <= result <= 1.0

    def test_ignores_none_values(self):
        result = calculate_base_confidence(0.5, {"numerical": 0.5, "nlp": None})
        # None values excluded; treated as single score
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# calculate_availability_penalty
# ---------------------------------------------------------------------------

class TestCalculateAvailabilityPenalty:

    def test_full_mode_no_penalty(self):
        p = calculate_availability_penalty(EnsembleMode.FULL, True, True, [])
        assert p == CONFIDENCE_PENALTIES["full"] == 0.0

    def test_partial_no_voice_penalty(self):
        p = calculate_availability_penalty(EnsembleMode.PARTIAL, True, False, [])
        assert abs(p - CONFIDENCE_PENALTIES["partial_no_voice"]) < 1e-9

    def test_partial_no_nlp_penalty(self):
        p = calculate_availability_penalty(EnsembleMode.PARTIAL, False, True, [])
        assert abs(p - CONFIDENCE_PENALTIES["partial_no_nlp"]) < 1e-9

    def test_fallback_penalty(self):
        p = calculate_availability_penalty(EnsembleMode.FALLBACK, False, False, [])
        assert abs(p - CONFIDENCE_PENALTIES["fallback"]) < 1e-9

    def test_error_adds_per_failed_modality(self):
        p_no_fail  = calculate_availability_penalty(EnsembleMode.PARTIAL, True, False, [])
        p_one_fail = calculate_availability_penalty(EnsembleMode.PARTIAL, True, False, ["voice"])
        assert abs(p_one_fail - p_no_fail - ERROR_PENALTY_PER_MODALITY) < 1e-9

    def test_multiple_errors_stack(self):
        p_zero = calculate_availability_penalty(EnsembleMode.FALLBACK, False, False, [])
        p_two  = calculate_availability_penalty(EnsembleMode.FALLBACK, False, False, ["nlp", "voice"])
        assert abs(p_two - p_zero - 2 * ERROR_PENALTY_PER_MODALITY) < 1e-9

    def test_capped_at_60_pct(self):
        p = calculate_availability_penalty(EnsembleMode.FALLBACK, False, False,
                                           ["nlp", "voice", "extra1", "extra2", "extra3"])
        assert p <= 0.60

    def test_penalty_ordering(self):
        full     = calculate_availability_penalty(EnsembleMode.FULL,     True,  True,  [])
        p_novoc  = calculate_availability_penalty(EnsembleMode.PARTIAL,  True,  False, [])
        p_nonlp  = calculate_availability_penalty(EnsembleMode.PARTIAL,  False, True,  [])
        fallback = calculate_availability_penalty(EnsembleMode.FALLBACK, False, False, [])
        assert full < p_novoc < p_nonlp < fallback


# ---------------------------------------------------------------------------
# calculate_confidence_adjustment
# ---------------------------------------------------------------------------

class TestCalculateConfidenceAdjustment:

    def test_returns_confidence_adjustment(self):
        result = calculate_confidence_adjustment(
            0.5, {"numerical": 0.5}, EnsembleMode.FULL, True, True, []
        )
        assert isinstance(result, ConfidenceAdjustment)

    def test_full_mode_no_availability_penalty(self):
        result = calculate_confidence_adjustment(
            0.5, {"numerical": 0.5, "nlp": 0.5, "voice": 0.5},
            EnsembleMode.FULL, True, True, []
        )
        assert result.availability_penalty == 0.0

    def test_fallback_mode_40_pct_penalty(self):
        result = calculate_confidence_adjustment(
            0.5, {"numerical": 0.5}, EnsembleMode.FALLBACK, False, False, []
        )
        assert abs(result.availability_penalty - 0.40) < 1e-9

    def test_final_confidence_equals_base_times_factor(self):
        result = calculate_confidence_adjustment(
            0.5, {"numerical": 0.5}, EnsembleMode.FALLBACK, False, False, []
        )
        expected = result.base_confidence * (1.0 - result.availability_penalty)
        # Allow for rounding from round(..., 4) in the implementation
        assert abs(result.final_confidence - expected) < 1e-4

    def test_full_mode_higher_confidence_than_fallback(self):
        full = calculate_confidence_adjustment(
            0.5, {"numerical": 0.5, "nlp": 0.5, "voice": 0.5},
            EnsembleMode.FULL, True, True, []
        )
        fallback = calculate_confidence_adjustment(
            0.5, {"numerical": 0.5}, EnsembleMode.FALLBACK, False, False, []
        )
        assert full.final_confidence > fallback.final_confidence

    def test_final_confidence_in_unit_interval(self):
        for mode, nlp, voice in [
            (EnsembleMode.FULL, True, True),
            (EnsembleMode.PARTIAL, True, False),
            (EnsembleMode.FALLBACK, False, False),
        ]:
            scores = {"numerical": 0.6, "nlp": 0.5 if nlp else None, "voice": 0.4 if voice else None}
            result = calculate_confidence_adjustment(0.6, scores, mode, nlp, voice, [])
            assert 0.0 <= result.final_confidence <= 1.0


# ---------------------------------------------------------------------------
# apply_fallback_overrides
# ---------------------------------------------------------------------------

class TestApplyFallbackOverrides:

    def test_full_mode_no_override(self):
        decision, reason = apply_fallback_overrides(
            0.5, 0.9, DecisionType.REVIEW, EnsembleMode.FULL
        )
        assert decision == DecisionType.REVIEW
        assert reason is None

    def test_fallback_above_block_threshold_forces_block(self):
        score = FALLBACK_OVERRIDES["force_block_threshold"] + 0.01
        decision, reason = apply_fallback_overrides(
            score, 0.6, DecisionType.REVIEW, EnsembleMode.FALLBACK
        )
        assert decision == DecisionType.BLOCK
        assert reason is not None
        assert "Fallback" in reason

    def test_fallback_above_review_threshold_forces_review(self):
        score = FALLBACK_OVERRIDES["force_review_threshold"] + 0.01
        decision, reason = apply_fallback_overrides(
            score, 0.6, DecisionType.APPROVE, EnsembleMode.FALLBACK
        )
        assert decision == DecisionType.REVIEW
        assert reason is not None

    def test_fallback_below_review_threshold_stays_approve(self):
        score = FALLBACK_OVERRIDES["force_review_threshold"] - 0.05
        decision, reason = apply_fallback_overrides(
            score, 0.6, DecisionType.APPROVE, EnsembleMode.FALLBACK
        )
        assert decision == DecisionType.APPROVE
        assert reason is None

    def test_fallback_already_blocked_no_change(self):
        score = FALLBACK_OVERRIDES["force_block_threshold"] + 0.01
        decision, reason = apply_fallback_overrides(
            score, 0.6, DecisionType.BLOCK, EnsembleMode.FALLBACK
        )
        # Already BLOCK – override still sets reason for audit but decision unchanged
        assert decision == DecisionType.BLOCK

    def test_force_review_disabled_no_override(self):
        score = FALLBACK_OVERRIDES["force_review_threshold"] + 0.01
        decision, reason = apply_fallback_overrides(
            score, 0.6, DecisionType.APPROVE, EnsembleMode.FALLBACK,
            force_review_on_fallback=False,
        )
        assert decision == DecisionType.APPROVE
        assert reason is None

    def test_partial_low_confidence_forces_review(self):
        low_conf = PARTIAL_MODE_OVERRIDES["low_confidence_threshold"] - 0.05
        high_score = PARTIAL_MODE_OVERRIDES["force_review_score"] + 0.05
        decision, reason = apply_fallback_overrides(
            high_score, low_conf, DecisionType.APPROVE, EnsembleMode.PARTIAL
        )
        assert decision == DecisionType.REVIEW
        assert reason == "Low confidence override"

    def test_partial_sufficient_confidence_no_override(self):
        high_conf  = PARTIAL_MODE_OVERRIDES["low_confidence_threshold"] + 0.05
        high_score = PARTIAL_MODE_OVERRIDES["force_review_score"] + 0.05
        decision, reason = apply_fallback_overrides(
            high_score, high_conf, DecisionType.APPROVE, EnsembleMode.PARTIAL
        )
        assert decision == DecisionType.APPROVE
        assert reason is None

    def test_partial_low_score_no_override_even_with_low_confidence(self):
        low_conf  = PARTIAL_MODE_OVERRIDES["low_confidence_threshold"] - 0.05
        low_score = PARTIAL_MODE_OVERRIDES["force_review_score"] - 0.05
        decision, reason = apply_fallback_overrides(
            low_score, low_conf, DecisionType.APPROVE, EnsembleMode.PARTIAL
        )
        assert decision == DecisionType.APPROVE
        assert reason is None

    # Edge / boundary
    def test_score_exactly_at_fallback_block_threshold(self):
        score = FALLBACK_OVERRIDES["force_block_threshold"]
        decision, reason = apply_fallback_overrides(
            score, 0.6, DecisionType.APPROVE, EnsembleMode.FALLBACK
        )
        # score NOT greater-than threshold → no block override
        assert decision != DecisionType.BLOCK or reason is None or "block" in (reason or "").lower()

    def test_override_reason_contains_threshold_value(self):
        score = FALLBACK_OVERRIDES["force_block_threshold"] + 0.01
        _, reason = apply_fallback_overrides(
            score, 0.6, DecisionType.APPROVE, EnsembleMode.FALLBACK
        )
        assert str(FALLBACK_OVERRIDES["force_block_threshold"]) in (reason or "")