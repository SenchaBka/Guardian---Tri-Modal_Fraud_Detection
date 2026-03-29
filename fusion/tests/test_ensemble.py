"""
Unit Tests – fusion.ensemble
==============================
Covers renormalize_weights, compute_fraud_score, determine_mode,
determine_decision, determine_risk_level, get_modality_combination_info.
"""

import pytest
from fusion.ensemble import (
    renormalize_weights,
    compute_fraud_score,
    determine_mode,
    determine_decision,
    determine_risk_level,
    get_modality_combination_info,
)
from fusion.schemas import EnsembleMode, DecisionType, RiskLevel
from fusion.config import (
    BASE_WEIGHTS, PRECALCULATED_WEIGHTS,
    DECISION_THRESHOLDS, RISK_LEVEL_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# renormalize_weights
# ---------------------------------------------------------------------------

class TestRenormalizeWeights:

    def test_full_mode_matches_precalculated(self):
        w = renormalize_weights(True, True, True)
        assert w == PRECALCULATED_WEIGHTS[(True, True, True)]

    def test_full_mode_weights_sum_to_one(self):
        w = renormalize_weights(True, True, True)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_full_mode_base_weights_unchanged(self):
        w = renormalize_weights(True, True, True)
        assert abs(w["numerical"] - 0.45) < 1e-9
        assert abs(w["nlp"]       - 0.35) < 1e-9
        assert abs(w["voice"]     - 0.20) < 1e-9

    def test_partial_no_voice_matches_precalculated(self):
        w = renormalize_weights(True, True, False)
        assert w == PRECALCULATED_WEIGHTS[(True, True, False)]

    def test_partial_no_voice_voice_weight_is_zero(self):
        w = renormalize_weights(True, True, False)
        assert w["voice"] == 0.0

    def test_partial_no_voice_weights_sum_to_one(self):
        w = renormalize_weights(True, True, False)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_partial_no_voice_ratio_preserved(self):
        w = renormalize_weights(True, True, False)
        expected_ratio = BASE_WEIGHTS["numerical"] / BASE_WEIGHTS["nlp"]
        assert abs(w["numerical"] / w["nlp"] - expected_ratio) < 1e-6

    def test_partial_no_nlp_matches_precalculated(self):
        w = renormalize_weights(True, False, True)
        assert w == PRECALCULATED_WEIGHTS[(True, False, True)]

    def test_partial_no_nlp_nlp_weight_is_zero(self):
        w = renormalize_weights(True, False, True)
        assert w["nlp"] == 0.0

    def test_partial_no_nlp_weights_sum_to_one(self):
        w = renormalize_weights(True, False, True)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_fallback_numerical_weight_is_one(self):
        w = renormalize_weights(True, False, False)
        assert abs(w["numerical"] - 1.0) < 1e-9

    def test_fallback_optional_weights_are_zero(self):
        w = renormalize_weights(True, False, False)
        assert w["nlp"]   == 0.0
        assert w["voice"] == 0.0

    def test_fallback_weights_sum_to_one(self):
        w = renormalize_weights(True, False, False)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_numerical_false_raises_value_error(self):
        with pytest.raises(ValueError, match="required"):
            renormalize_weights(False, True, True)

    def test_returns_copy_not_reference(self):
        w1 = renormalize_weights(True, True, True)
        w2 = renormalize_weights(True, True, True)
        w1["numerical"] = 0.0
        assert w2["numerical"] == 0.45  # mutation does not affect future calls

    def test_numerical_always_dominant(self):
        for nlp, voice in [(True, True), (True, False), (False, True)]:
            w = renormalize_weights(True, nlp, voice)
            assert w["numerical"] >= w["nlp"]
            assert w["numerical"] >= w["voice"]


# ---------------------------------------------------------------------------
# compute_fraud_score
# ---------------------------------------------------------------------------

class TestComputeFraudScore:

    def test_full_mode_weighted_average(self):
        weights = {"numerical": 0.45, "nlp": 0.35, "voice": 0.20}
        scores  = {"numerical": 0.8,  "nlp": 0.6,  "voice": 0.4}
        expected = 0.45*0.8 + 0.35*0.6 + 0.20*0.4
        assert abs(compute_fraud_score(scores, weights) - expected) < 1e-9

    def test_ignores_none_scores(self):
        weights = {"numerical": 0.5625, "nlp": 0.4375, "voice": 0.0}
        scores  = {"numerical": 0.7,    "nlp": 0.5,    "voice": None}
        expected = 0.5625*0.7 + 0.4375*0.5
        assert abs(compute_fraud_score(scores, weights) - expected) < 1e-9

    def test_fallback_equals_numerical_score(self):
        weights = {"numerical": 1.0, "nlp": 0.0, "voice": 0.0}
        scores  = {"numerical": 0.65, "nlp": None, "voice": None}
        assert abs(compute_fraud_score(scores, weights) - 0.65) < 1e-9

    def test_zero_score_returns_zero(self):
        weights = {"numerical": 1.0, "nlp": 0.0, "voice": 0.0}
        scores  = {"numerical": 0.0, "nlp": None, "voice": None}
        assert compute_fraud_score(scores, weights) == 0.0

    def test_max_score_clamped_to_one(self):
        weights = {"numerical": 0.45, "nlp": 0.35, "voice": 0.20}
        scores  = {"numerical": 1.0,  "nlp": 1.0,  "voice": 1.0}
        assert compute_fraud_score(scores, weights) <= 1.0

    def test_output_always_in_unit_interval(self):
        for n, nlp, v in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.3, 0.8)]:
            w = renormalize_weights(True, True, True)
            s = {"numerical": n, "nlp": nlp, "voice": v}
            result = compute_fraud_score(s, w)
            assert 0.0 <= result <= 1.0

    def test_unknown_modality_in_scores_ignored(self):
        weights = {"numerical": 1.0}
        scores  = {"numerical": 0.5, "unknown_stream": 0.9}
        assert abs(compute_fraud_score(scores, weights) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# determine_mode
# ---------------------------------------------------------------------------

class TestDetermineMode:

    def test_all_three_is_full(self):
        assert determine_mode(True, True, True) == EnsembleMode.FULL

    def test_numerical_nlp_is_partial(self):
        assert determine_mode(True, True, False) == EnsembleMode.PARTIAL

    def test_numerical_voice_is_partial(self):
        assert determine_mode(True, False, True) == EnsembleMode.PARTIAL

    def test_numerical_only_is_fallback(self):
        assert determine_mode(True, False, False) == EnsembleMode.FALLBACK

    def test_numerical_false_raises(self):
        with pytest.raises(ValueError):
            determine_mode(False, True, True)


# ---------------------------------------------------------------------------
# determine_decision
# ---------------------------------------------------------------------------

class TestDetermineDecision:

    def test_zero_is_approve(self):
        assert determine_decision(0.0) == DecisionType.APPROVE

    def test_at_approve_ceiling_is_approve(self):
        assert determine_decision(DECISION_THRESHOLDS["approve"][1]) == DecisionType.APPROVE

    def test_just_above_approve_ceiling_is_review(self):
        assert determine_decision(DECISION_THRESHOLDS["approve"][1] + 0.01) == DecisionType.REVIEW

    def test_mid_range_is_review(self):
        assert determine_decision(0.5) == DecisionType.REVIEW

    def test_at_review_ceiling_is_review(self):
        assert determine_decision(DECISION_THRESHOLDS["review"][1]) == DecisionType.REVIEW

    def test_just_above_review_ceiling_is_block(self):
        assert determine_decision(DECISION_THRESHOLDS["review"][1] + 0.01) == DecisionType.BLOCK

    def test_one_is_block(self):
        assert determine_decision(1.0) == DecisionType.BLOCK

    def test_threshold_override_block(self):
        # score 0.9 > override 0.8 → block
        assert determine_decision(0.9, threshold_override=0.8) == DecisionType.BLOCK

    def test_threshold_override_review(self):
        # score 0.5 > 0.8 * 0.5 = 0.4 → review
        assert determine_decision(0.5, threshold_override=0.8) == DecisionType.REVIEW

    def test_threshold_override_approve(self):
        # score 0.1 < 0.8 * 0.5 = 0.4 → approve
        assert determine_decision(0.1, threshold_override=0.8) == DecisionType.APPROVE


# ---------------------------------------------------------------------------
# determine_risk_level
# ---------------------------------------------------------------------------

class TestDetermineRiskLevel:

    def test_zero_is_low(self):
        assert determine_risk_level(0.0) == RiskLevel.LOW

    def test_at_low_ceiling_is_low(self):
        assert determine_risk_level(RISK_LEVEL_THRESHOLDS["low"][1]) == RiskLevel.LOW

    def test_medium_range(self):
        assert determine_risk_level(0.40) == RiskLevel.MEDIUM

    def test_high_range(self):
        assert determine_risk_level(0.65) == RiskLevel.HIGH

    def test_one_is_critical(self):
        assert determine_risk_level(1.0) == RiskLevel.CRITICAL

    def test_critical_range(self):
        assert determine_risk_level(0.80) == RiskLevel.CRITICAL

    def test_ordering(self):
        levels = [determine_risk_level(s) for s in [0.1, 0.35, 0.65, 0.9]]
        assert levels == [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


# ---------------------------------------------------------------------------
# get_modality_combination_info
# ---------------------------------------------------------------------------

class TestGetModalityCombinationInfo:

    def test_full_returns_100_pct(self):
        desc, conf = get_modality_combination_info(True, True)
        assert conf == 1.00
        assert "Numerical + NLP + Voice" in desc

    def test_no_voice_returns_85_pct(self):
        _, conf = get_modality_combination_info(True, False)
        assert conf == 0.85

    def test_no_nlp_returns_80_pct(self):
        _, conf = get_modality_combination_info(False, True)
        assert conf == 0.80

    def test_fallback_returns_60_pct(self):
        desc, conf = get_modality_combination_info(False, False)
        assert conf == 0.60
        assert "Fallback" in desc