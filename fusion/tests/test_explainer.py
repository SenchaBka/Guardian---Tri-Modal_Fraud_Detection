"""
Unit Tests – fusion.explainer
===============================
Covers calculate_shap_contributions, extract_top_factors,
generate_narrative, generate_missing_modality_note, generate_explanation.
"""

import pytest
from fusion.explainer import (
    calculate_shap_contributions,
    extract_top_factors,
    generate_narrative,
    generate_missing_modality_note,
    generate_explanation,
)
from fusion.schemas import EnsembleMode, ShapValues, Explanation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FULL_SCORES   = {"numerical": 0.8, "nlp": 0.6, "voice": 0.3}
FULL_WEIGHTS  = {"numerical": 0.45, "nlp": 0.35, "voice": 0.20}
FALLBACK_SCORES  = {"numerical": 0.7, "nlp": None, "voice": None}
FALLBACK_WEIGHTS = {"numerical": 1.0, "nlp": 0.0, "voice": 0.0}

NUMERICAL_SIGNALS = {
    "amount_anomaly":    0.85,
    "velocity_risk":     0.55,
    "pattern_deviation": 0.60,
    "geo_risk":          0.40,
}
NLP_SIGNALS = {
    "semantic_risk":        0.70,
    "typosquatting_risk":   0.20,
    "entity_inconsistency": 0.50,
}
VOICE_SIGNALS = {
    "deepfake_probability":    0.90,
    "voice_stress_indicator":  0.65,
    "speaker_consistency":     0.30,
}

ALL_SIGNALS  = {"numerical": NUMERICAL_SIGNALS, "nlp": NLP_SIGNALS, "voice": VOICE_SIGNALS}
NONE_SIGNALS = {"numerical": NUMERICAL_SIGNALS, "nlp": {},          "voice": {}}


# ---------------------------------------------------------------------------
# calculate_shap_contributions
# ---------------------------------------------------------------------------

class TestCalculateShapContributions:

    def test_returns_shap_values(self):
        result = calculate_shap_contributions(FULL_SCORES, FULL_WEIGHTS, 0.65)
        assert isinstance(result, ShapValues)

    def test_numerical_contribution_nonzero(self):
        result = calculate_shap_contributions(FULL_SCORES, FULL_WEIGHTS, 0.65)
        assert result.numerical_contribution != 0.0

    def test_nlp_contribution_present_when_available(self):
        result = calculate_shap_contributions(FULL_SCORES, FULL_WEIGHTS, 0.65)
        assert result.nlp_contribution is not None

    def test_voice_contribution_present_when_available(self):
        result = calculate_shap_contributions(FULL_SCORES, FULL_WEIGHTS, 0.65)
        assert result.voice_contribution is not None

    def test_nlp_contribution_none_when_unavailable(self):
        result = calculate_shap_contributions(FALLBACK_SCORES, FALLBACK_WEIGHTS, 0.7)
        assert result.nlp_contribution is None

    def test_voice_contribution_none_when_unavailable(self):
        result = calculate_shap_contributions(FALLBACK_SCORES, FALLBACK_WEIGHTS, 0.7)
        assert result.voice_contribution is None

    def test_shap_contribution_formula(self):
        from fusion.config import SHAP_BASELINE_RATE
        result = calculate_shap_contributions(
            {"numerical": 0.8, "nlp": None, "voice": None},
            {"numerical": 1.0, "nlp": 0.0, "voice": 0.0},
            0.8,
        )
        expected = round(1.0 * (0.8 - SHAP_BASELINE_RATE), 4)
        assert abs(result.numerical_contribution - expected) < 1e-6

    def test_zero_weight_gives_none(self):
        result = calculate_shap_contributions(
            {"numerical": 0.5, "nlp": 0.5, "voice": 0.5},
            {"numerical": 1.0, "nlp": 0.0, "voice": 0.0},
            0.5,
        )
        assert result.nlp_contribution   is None
        assert result.voice_contribution is None


# ---------------------------------------------------------------------------
# extract_top_factors
# ---------------------------------------------------------------------------

class TestExtractTopFactors:

    def test_returns_list(self):
        factors = extract_top_factors(FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS)
        assert isinstance(factors, list)

    def test_limited_to_top_factors_count(self):
        from fusion.config import TOP_FACTORS_COUNT
        factors = extract_top_factors(FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS)
        assert len(factors) <= TOP_FACTORS_COUNT

    def test_sorted_by_impact_descending(self):
        factors = extract_top_factors(FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS)
        impacts = [f.impact for f in factors]
        assert impacts == sorted(impacts, reverse=True)

    def test_direction_positive_for_high_signal(self):
        # deepfake_probability = 0.90 → direction should be positive (> 0.5)
        factors = extract_top_factors(FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS)
        deepfake = next((f for f in factors if "Deepfake" in f.feature), None)
        if deepfake:
            assert deepfake.direction == "positive"

    def test_direction_negative_for_low_signal(self):
        # typosquatting_risk = 0.20 → direction negative (< 0.5)
        factors = extract_top_factors(FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS)
        typo = next((f for f in factors if "Typosquatting" in f.feature), None)
        if typo:
            assert typo.direction == "negative"

    def test_skips_zero_weight_modalities(self):
        # nlp and voice have weight 0 in fallback → no nlp/voice factors
        factors = extract_top_factors(FALLBACK_SCORES, ALL_SIGNALS, FALLBACK_WEIGHTS)
        for f in factors:
            assert "Semantic" not in f.feature
            assert "Deepfake" not in f.feature

    def test_empty_signals_returns_empty_list(self):
        factors = extract_top_factors(
            {"numerical": None, "nlp": None, "voice": None}, {}, FULL_WEIGHTS
        )
        assert factors == []

    def test_display_name_mapping_applied(self):
        factors = extract_top_factors(FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS)
        raw_names = {"amount_anomaly", "velocity_risk", "semantic_risk"}
        factor_names = {f.feature for f in factors}
        # Raw keys should have been mapped to display names
        for raw in raw_names:
            assert raw not in factor_names

    def test_custom_num_factors_respected(self):
        factors = extract_top_factors(FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS, num_factors=2)
        assert len(factors) <= 2


# ---------------------------------------------------------------------------
# generate_narrative
# ---------------------------------------------------------------------------

class TestGenerateNarrative:

    def test_low_score_says_low_risk(self):
        narrative = generate_narrative(0.1, EnsembleMode.FULL, [], FULL_SCORES)
        assert "Low fraud risk" in narrative

    def test_moderate_score_says_moderate(self):
        narrative = generate_narrative(0.4, EnsembleMode.FULL, [], FULL_SCORES)
        assert "Moderate fraud risk" in narrative

    def test_elevated_score_says_elevated(self):
        narrative = generate_narrative(0.6, EnsembleMode.FULL, [], FULL_SCORES)
        assert "Elevated fraud risk" in narrative

    def test_high_score_says_high(self):
        narrative = generate_narrative(0.85, EnsembleMode.FULL, [], FULL_SCORES)
        assert "High fraud risk" in narrative

    def test_full_mode_mentions_tri_modal(self):
        narrative = generate_narrative(0.5, EnsembleMode.FULL, [], FULL_SCORES)
        assert "tri-modal" in narrative.lower() or "full" in narrative.lower()

    def test_fallback_mode_mentions_numerical_only(self):
        narrative = generate_narrative(0.5, EnsembleMode.FALLBACK, [], FALLBACK_SCORES)
        assert "Numerical only" in narrative or "fallback" in narrative.lower()

    def test_no_factors_says_no_significant(self):
        narrative = generate_narrative(0.5, EnsembleMode.FULL, [], FULL_SCORES)
        assert "No significant risk factors identified." in narrative

    def test_with_factors_lists_them(self):
        from fusion.schemas import TopFactor
        factors = [TopFactor(feature="Geographic Risk", impact=0.35, direction="positive")]
        narrative = generate_narrative(0.5, EnsembleMode.FULL, factors, FULL_SCORES)
        assert "Geographic Risk" in narrative

    def test_returns_string(self):
        assert isinstance(generate_narrative(0.5, EnsembleMode.FULL, [], FULL_SCORES), str)


# ---------------------------------------------------------------------------
# generate_missing_modality_note
# ---------------------------------------------------------------------------

class TestGenerateMissingModalityNote:

    def test_all_available_no_failed_returns_none(self):
        assert generate_missing_modality_note(True, True, []) is None

    def test_voice_unavailable_mentions_voice(self):
        note = generate_missing_modality_note(True, False, [])
        assert note is not None
        assert "Voice" in note

    def test_nlp_unavailable_mentions_nlp(self):
        note = generate_missing_modality_note(False, True, [])
        assert note is not None
        assert "NLP" in note

    def test_both_unavailable_mentions_both(self):
        note = generate_missing_modality_note(False, False, [])
        assert note is not None
        assert "Voice" in note and "NLP" in note

    def test_failed_modality_excluded_from_used_list(self):
        note = generate_missing_modality_note(True, True, ["nlp"])
        # NLP failed at runtime, so it should not be listed as used
        assert note is not None
        assert "text analysis" not in (note or "").lower()

    def test_note_mentions_what_was_used(self):
        note = generate_missing_modality_note(True, False, [])
        assert "transaction patterns" in note


# ---------------------------------------------------------------------------
# generate_explanation (integration)
# ---------------------------------------------------------------------------

class TestGenerateExplanation:

    def test_returns_explanation(self):
        result = generate_explanation(
            0.65, FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS,
            EnsembleMode.FULL, True, True, [],
        )
        assert isinstance(result, Explanation)

    def test_full_mode_no_missing_note(self):
        result = generate_explanation(
            0.65, FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS,
            EnsembleMode.FULL, True, True, [],
        )
        assert result.missing_modality_note is None

    def test_fallback_has_missing_note(self):
        result = generate_explanation(
            0.5, FALLBACK_SCORES, NONE_SIGNALS, FALLBACK_WEIGHTS,
            EnsembleMode.FALLBACK, False, False, [],
        )
        assert result.missing_modality_note is not None

    def test_narrative_is_non_empty_string(self):
        result = generate_explanation(
            0.5, FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS,
            EnsembleMode.FULL, True, True, [],
        )
        assert isinstance(result.narrative, str) and len(result.narrative) > 0

    def test_shap_values_populated(self):
        result = generate_explanation(
            0.65, FULL_SCORES, ALL_SIGNALS, FULL_WEIGHTS,
            EnsembleMode.FULL, True, True, [],
        )
        assert isinstance(result.shap_values, ShapValues)
        assert result.shap_values.numerical_contribution != 0.0