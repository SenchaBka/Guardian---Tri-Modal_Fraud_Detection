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
    
    When modalities are missing, their weight is redistributed 
    proportionally to remaining modalities so weights sum to 1.0.
    
    Args:
        numerical_available: Whether numerical stream is available (must be True)
        nlp_available: Whether NLP stream is available
        voice_available: Whether voice stream is available
    
    Returns:
        Dict mapping modality names to their renormalized weights.
    """
    if not numerical_available:
        raise ValueError("Numerical modality is required and must always be available")
    
    key = (numerical_available, nlp_available, voice_available)
    if key in PRECALCULATED_WEIGHTS:
        return PRECALCULATED_WEIGHTS[key].copy()
    
    weights = {}
    available_sum = 0.0
    
    if numerical_available:
        available_sum += BASE_WEIGHTS["numerical"]
    if nlp_available:
        available_sum += BASE_WEIGHTS["nlp"]
    if voice_available:
        available_sum += BASE_WEIGHTS["voice"]
    
    weights["numerical"] = BASE_WEIGHTS["numerical"] / available_sum if numerical_available else 0.0
    weights["nlp"] = BASE_WEIGHTS["nlp"] / available_sum if nlp_available else 0.0
    weights["voice"] = BASE_WEIGHTS["voice"] / available_sum if voice_available else 0.0
    
    return weights


def determine_mode(
    numerical_used: bool,
    nlp_used: bool,
    voice_used: bool,
) -> EnsembleMode:
    """
    Determine the ensemble operating mode based on which modalities were used.
    
    Args:
        numerical_used: Whether numerical contributed to score
        nlp_used: Whether NLP contributed to score  
        voice_used: Whether voice contributed to score
    
    Returns:
        EnsembleMode enum indicating the operating mode
    """
    if not numerical_used:
        raise ValueError("Numerical modality must always be used")
    
    modalities_used = sum([numerical_used, nlp_used, voice_used])
    
    if modalities_used == 3:
        return EnsembleMode.FULL
    elif modalities_used == 2:
        return EnsembleMode.PARTIAL
    else:
        return EnsembleMode.FALLBACK


def compute_fraud_score(
    scores: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> float:
    """
    Compute the final weighted fraud score from modality scores.
    
    Args:
        scores: Dict mapping modality names to their risk scores (0.0-1.0).
        weights: Renormalized weights from renormalize_weights()
    
    Returns:
        Combined fraud score in range [0.0, 1.0]
    """
    fraud_score = 0.0
    
    for modality, score in scores.items():
        if score is not None and modality in weights:
            fraud_score += weights[modality] * score
    
    return max(0.0, min(1.0, fraud_score))


def determine_decision(
    fraud_score: float,
    threshold_override: Optional[float] = None,
) -> DecisionType:
    """
    Determine the categorical decision based on fraud score.
    
    Args:
        fraud_score: Combined fraud score from compute_fraud_score()
        threshold_override: Optional custom threshold for review->block boundary
    
    Returns:
        DecisionType enum (APPROVE, REVIEW, or BLOCK)
    """
    if threshold_override is not None:
        if fraud_score > threshold_override:
            return DecisionType.BLOCK
        elif fraud_score > threshold_override * 0.5:
            return DecisionType.REVIEW
        else:
            return DecisionType.APPROVE
    
    if fraud_score <= DECISION_THRESHOLDS["approve"][1]:
        return DecisionType.APPROVE
    elif fraud_score <= DECISION_THRESHOLDS["review"][1]:
        return DecisionType.REVIEW
    else:
        return DecisionType.BLOCK


def determine_risk_level(fraud_score: float) -> RiskLevel:
    """
    Categorize the fraud score into a risk level for reporting.
    
    Args:
        fraud_score: Combined fraud score from compute_fraud_score()
    
    Returns:
        RiskLevel enum for dashboard display and prioritization
    """
    if fraud_score <= RISK_LEVEL_THRESHOLDS["low"][1]:
        return RiskLevel.LOW
    elif fraud_score <= RISK_LEVEL_THRESHOLDS["medium"][1]:
        return RiskLevel.MEDIUM
    elif fraud_score <= RISK_LEVEL_THRESHOLDS["high"][1]:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


def get_modality_combination_info(
    nlp_available: bool,
    voice_available: bool,
) -> Tuple[str, float]:
    """
    Get descriptive info about the modality combination being used.
    
    Args:
        nlp_available: Whether NLP is available
        voice_available: Whether voice is available
    
    Returns:
        Tuple of (description string, base confidence percentage)
    """
    if nlp_available and voice_available:
        return ("Full tri-modal ensemble: Numerical + NLP + Voice", 1.00)
    elif nlp_available and not voice_available:
        return ("Partial ensemble: Numerical + NLP (voice unavailable)", 0.85)
    elif not nlp_available and voice_available:
        return ("Partial ensemble: Numerical + Voice (NLP unavailable)", 0.80)
    else:
        return ("Fallback mode: Numerical only", 0.60)