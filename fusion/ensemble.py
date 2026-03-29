"""
Guardian Fusion Layer - Ensemble Logic
=======================================
Author: Sherwayne (ML Systems Architect)
"""

from typing import Dict, Optional, Tuple
from .config import (
    BASE_WEIGHTS,
    PRECALCULATED_WEIGHTS,
    DECISION_THRESHOLDS,
    RISK_LEVEL_THRESHOLDS,
)
from .schemas import EnsembleMode, DecisionType, RiskLevel


def renormalize_weights(
    numerical_available: bool = True,
    nlp_available: bool = True,
    voice_available: bool = True,
) -> Dict[str, float]:
    """
    Renormalize ensemble weights based on available modalities.

    When modalities are missing their weight is redistributed
    proportionally to remaining modalities so weights always sum to 1.0.
    """
    if not numerical_available:
        raise ValueError("Numerical modality is required and must always be available")

    key = (numerical_available, nlp_available, voice_available)
    if key in PRECALCULATED_WEIGHTS:
        return PRECALCULATED_WEIGHTS[key].copy()

    available_sum = 0.0
    if numerical_available: available_sum += BASE_WEIGHTS["numerical"]
    if nlp_available:        available_sum += BASE_WEIGHTS["nlp"]
    if voice_available:      available_sum += BASE_WEIGHTS["voice"]

    return {
        "numerical": BASE_WEIGHTS["numerical"] / available_sum if numerical_available else 0.0,
        "nlp":       BASE_WEIGHTS["nlp"]       / available_sum if nlp_available       else 0.0,
        "voice":     BASE_WEIGHTS["voice"]     / available_sum if voice_available      else 0.0,
    }


def determine_mode(
    numerical_used: bool,
    nlp_used: bool,
    voice_used: bool,
) -> EnsembleMode:
    """Determine the ensemble operating mode based on which modalities were used."""
    if not numerical_used:
        raise ValueError("Numerical modality must always be used")

    count = sum([numerical_used, nlp_used, voice_used])
    if count == 3:   return EnsembleMode.FULL
    elif count == 2: return EnsembleMode.PARTIAL
    else:            return EnsembleMode.FALLBACK


def compute_fraud_score(
    scores: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> float:
    """Compute the final weighted fraud score from modality scores."""
    fraud_score = 0.0
    for modality, score in scores.items():
        if score is not None and modality in weights:
            fraud_score += weights[modality] * score
    return max(0.0, min(1.0, fraud_score))


def determine_decision(
    fraud_score: float,
    threshold_override: Optional[float] = None,
) -> DecisionType:
    """Determine the categorical decision based on fraud score."""
    if threshold_override is not None:
        if fraud_score > threshold_override:                return DecisionType.BLOCK
        elif fraud_score > threshold_override * 0.5:       return DecisionType.REVIEW
        else:                                               return DecisionType.APPROVE

    if fraud_score <= DECISION_THRESHOLDS["approve"][1]:   return DecisionType.APPROVE
    elif fraud_score <= DECISION_THRESHOLDS["review"][1]:  return DecisionType.REVIEW
    else:                                                   return DecisionType.BLOCK


def determine_risk_level(fraud_score: float) -> RiskLevel:
    """Categorize the fraud score into a risk level for reporting."""
    if fraud_score <= RISK_LEVEL_THRESHOLDS["low"][1]:      return RiskLevel.LOW
    elif fraud_score <= RISK_LEVEL_THRESHOLDS["medium"][1]: return RiskLevel.MEDIUM
    elif fraud_score <= RISK_LEVEL_THRESHOLDS["high"][1]:   return RiskLevel.HIGH
    else:                                                    return RiskLevel.CRITICAL


def get_modality_combination_info(
    nlp_available: bool,
    voice_available: bool,
) -> Tuple[str, float]:
    """Get descriptive info about the modality combination being used."""
    if nlp_available and voice_available:
        return ("Full tri-modal ensemble: Numerical + NLP + Voice", 1.00)
    elif nlp_available and not voice_available:
        return ("Partial ensemble: Numerical + NLP (voice unavailable)", 0.85)
    elif not nlp_available and voice_available:
        return ("Partial ensemble: Numerical + Voice (NLP unavailable)", 0.80)
    else:
        return ("Fallback mode: Numerical only", 0.60)