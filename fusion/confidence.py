"""
Guardian Fusion Layer - Confidence Adjustment Logic
====================================================

Author: Sherwayne (ML Systems Architect)
"""

from typing import Dict, Optional, List, Tuple
from .schemas import EnsembleMode, DecisionType, ConfidenceAdjustment
from .config import (
    CONFIDENCE_PENALTIES,
    ERROR_PENALTY_PER_MODALITY,
    FALLBACK_OVERRIDES,
    PARTIAL_MODE_OVERRIDES,
)


def calculate_base_confidence(
    fraud_score: float,
    modality_scores: Dict[str, Optional[float]],
) -> float:
    """Calculate the base (raw) confidence before modality penalties."""
    available_scores = [s for s in modality_scores.values() if s is not None]
    
    if len(available_scores) == 0:
        return 0.5
    
    if len(available_scores) > 1:
        mean_score = sum(available_scores) / len(available_scores)
        variance = sum((s - mean_score) ** 2 for s in available_scores) / len(available_scores)
        agreement_confidence = 1.0 - min(1.0, variance * 4)
    else:
        agreement_confidence = 0.7
    
    boundaries = [0.30, 0.70]
    min_boundary_distance = min(abs(fraud_score - b) for b in boundaries)
    boundary_confidence = min(1.0, min_boundary_distance / 0.35)
    
    base_confidence = (agreement_confidence * 0.6) + (boundary_confidence * 0.4)
    
    return round(base_confidence, 4)


def calculate_availability_penalty(
    ensemble_mode: EnsembleMode,
    nlp_available: bool,
    voice_available: bool,
    modalities_failed: List[str],
) -> float:
    """Calculate the confidence penalty based on modality availability."""
    if ensemble_mode == EnsembleMode.FULL:
        base_penalty = CONFIDENCE_PENALTIES["full"]
    elif ensemble_mode == EnsembleMode.PARTIAL:
        if not voice_available:
            base_penalty = CONFIDENCE_PENALTIES["partial_no_voice"]
        else:
            base_penalty = CONFIDENCE_PENALTIES["partial_no_nlp"]
    else:
        base_penalty = CONFIDENCE_PENALTIES["fallback"]
    
    error_penalty = len(modalities_failed) * ERROR_PENALTY_PER_MODALITY
    total_penalty = min(0.60, base_penalty + error_penalty)
    
    return round(total_penalty, 4)


def calculate_confidence_adjustment(
    fraud_score: float,
    modality_scores: Dict[str, Optional[float]],
    ensemble_mode: EnsembleMode,
    nlp_available: bool,
    voice_available: bool,
    modalities_failed: List[str],
) -> ConfidenceAdjustment:
    """Calculate the full confidence adjustment with transparency."""
    base_confidence = calculate_base_confidence(fraud_score, modality_scores)
    
    penalty = calculate_availability_penalty(
        ensemble_mode, nlp_available, voice_available, modalities_failed
    )
    
    final_confidence = base_confidence * (1.0 - penalty)
    
    return ConfidenceAdjustment(
        base_confidence=round(base_confidence, 4),
        availability_penalty=round(penalty, 4),
        final_confidence=round(final_confidence, 4),
    )


def apply_fallback_overrides(
    fraud_score: float,
    confidence: float,
    decision: DecisionType,
    ensemble_mode: EnsembleMode,
    force_review_on_fallback: bool = True,
) -> Tuple[DecisionType, Optional[str]]:
    """Apply conservative overrides based on ensemble mode and confidence."""
    override_reason = None
    final_decision = decision
    
    if ensemble_mode == EnsembleMode.FALLBACK and force_review_on_fallback:
        if fraud_score > FALLBACK_OVERRIDES["force_block_threshold"]:
            if decision != DecisionType.BLOCK:
                final_decision = DecisionType.BLOCK
                override_reason = f"Fallback mode override: score > {FALLBACK_OVERRIDES['force_block_threshold']}"
        elif fraud_score > FALLBACK_OVERRIDES["force_review_threshold"]:
            if decision == DecisionType.APPROVE:
                final_decision = DecisionType.REVIEW
                override_reason = f"Fallback mode override: score > {FALLBACK_OVERRIDES['force_review_threshold']}"
    
    elif ensemble_mode == EnsembleMode.PARTIAL:
        if (confidence < PARTIAL_MODE_OVERRIDES["low_confidence_threshold"] and 
            fraud_score > PARTIAL_MODE_OVERRIDES["force_review_score"]):
            if decision == DecisionType.APPROVE:
                final_decision = DecisionType.REVIEW
                override_reason = "Low confidence override"
    
    return final_decision, override_reason