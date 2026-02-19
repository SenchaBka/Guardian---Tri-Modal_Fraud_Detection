"""
Guardian Fusion Layer - Explainable AI (XAI) Integration
=========================================================

Author: Sherwayne (ML Systems Architect)
"""

from typing import Dict, Optional, List
from .schemas import Explanation, TopFactor, ShapValues, EnsembleMode
from .config import TOP_FACTORS_COUNT, SHAP_BASELINE_RATE


def calculate_shap_contributions(
    modality_scores: Dict[str, Optional[float]],
    weights: Dict[str, float],
    fraud_score: float,
) -> ShapValues:
    """Calculate SHAP-style contribution values for each modality."""
    def calc_contribution(modality: str) -> Optional[float]:
        score = modality_scores.get(modality)
        weight = weights.get(modality, 0.0)
        
        if score is None or weight == 0:
            return None
        
        contribution = weight * (score - SHAP_BASELINE_RATE)
        return round(contribution, 4)
    
    return ShapValues(
        numerical_contribution=calc_contribution("numerical") or 0.0,
        nlp_contribution=calc_contribution("nlp"),
        voice_contribution=calc_contribution("voice"),
    )


def extract_top_factors(
    modality_scores: Dict[str, Optional[float]],
    modality_signals: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    num_factors: int = TOP_FACTORS_COUNT,
) -> List[TopFactor]:
    """Extract the top contributing factors across all modalities."""
    all_factors = []
    
    signal_display_names = {
        "amount_anomaly": "Transaction Amount Anomaly",
        "velocity_risk": "Transaction Velocity Risk",
        "pattern_deviation": "Behavioral Pattern Deviation",
        "geo_risk": "Geographic Risk",
        "semantic_risk": "Semantic Risk in Description",
        "typosquatting_risk": "Merchant Typosquatting Risk",
        "entity_inconsistency": "Entity Inconsistency",
        "deepfake_probability": "Deepfake Voice Detection",
        "voice_stress_indicator": "Voice Stress Level",
        "speaker_consistency": "Speaker Verification Score",
    }
    
    for modality, signals in modality_signals.items():
        weight = weights.get(modality, 0.0)
        
        if weight == 0 or signals is None:
            continue
        
        for signal_name, signal_value in signals.items():
            impact = weight * abs(signal_value - 0.5) * 2
            direction = "positive" if signal_value > 0.5 else "negative"
            display_name = signal_display_names.get(signal_name, signal_name)
            
            all_factors.append(TopFactor(
                feature=display_name,
                impact=round(impact, 4),
                direction=direction,
            ))
    
    all_factors.sort(key=lambda f: f.impact, reverse=True)
    return all_factors[:num_factors]


def generate_narrative(
    fraud_score: float,
    ensemble_mode: EnsembleMode,
    top_factors: List[TopFactor],
    modality_scores: Dict[str, Optional[float]],
) -> str:
    """Generate a human-readable explanation narrative."""
    if fraud_score <= 0.30:
        risk_desc = "Low fraud risk"
    elif fraud_score <= 0.50:
        risk_desc = "Moderate fraud risk"
    elif fraud_score <= 0.70:
        risk_desc = "Elevated fraud risk"
    else:
        risk_desc = "High fraud risk"
    
    if ensemble_mode == EnsembleMode.PARTIAL:
        used = []
        if modality_scores.get("numerical") is not None:
            used.append("Numerical")
        if modality_scores.get("nlp") is not None:
            used.append("NLP")
        if modality_scores.get("voice") is not None:
            used.append("Voice")
        mode_detail = f"partial ensemble ({' + '.join(used)})"
    elif ensemble_mode == EnsembleMode.FULL:
        mode_detail = "full tri-modal ensemble (Numerical + NLP + Voice)"
    else:
        mode_detail = "fallback mode (Numerical only)"
    
    if top_factors:
        def impact_level(impact: float) -> str:
            if impact > 0.3:
                return "strong"
            elif impact > 0.15:
                return "moderate"
            else:
                return "minor"
        
        factor_strs = [
            f"{f.feature} ({impact_level(f.impact)} {f.direction} impact)"
            for f in top_factors[:3]
        ]
        factors_desc = f"Primary factors: {', '.join(factor_strs)}."
    else:
        factors_desc = "No significant risk factors identified."
    
    return f"{risk_desc} ({fraud_score:.2f}) detected using {mode_detail}. {factors_desc}"


def generate_missing_modality_note(
    nlp_available: bool,
    voice_available: bool,
    modalities_failed: List[str],
) -> Optional[str]:
    """Generate context note about unavailable modalities."""
    if nlp_available and voice_available and not modalities_failed:
        return None
    
    missing = []
    if not voice_available:
        missing.append("Voice")
    if not nlp_available:
        missing.append("NLP")
    
    parts = []
    if missing:
        parts.append(f"{' and '.join(missing)} analysis unavailable for this transaction type.")
    
    used = ["transaction patterns (Numerical)"]
    if nlp_available and "nlp" not in modalities_failed:
        used.append("text analysis (NLP)")
    if voice_available and "voice" not in modalities_failed:
        used.append("voice analysis")
    
    parts.append(f"Decision based on {' and '.join(used)}.")
    
    return " ".join(parts)


def generate_explanation(
    fraud_score: float,
    modality_scores: Dict[str, Optional[float]],
    modality_signals: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    ensemble_mode: EnsembleMode,
    nlp_available: bool,
    voice_available: bool,
    modalities_failed: List[str],
) -> Explanation:
    """Generate the complete SHAP-based explanation for a fraud decision."""
    shap_values = calculate_shap_contributions(modality_scores, weights, fraud_score)
    top_factors = extract_top_factors(modality_scores, modality_signals, weights)
    narrative = generate_narrative(fraud_score, ensemble_mode, top_factors, modality_scores)
    missing_note = generate_missing_modality_note(nlp_available, voice_available, modalities_failed)
    
    return Explanation(
        top_factors=top_factors,
        shap_values=shap_values,
        narrative=narrative,
        missing_modality_note=missing_note,
    )