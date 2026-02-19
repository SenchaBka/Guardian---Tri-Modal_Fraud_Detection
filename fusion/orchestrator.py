"""
Guardian Fusion Layer - Request Orchestrator
=============================================

Author: Sherwayne (ML Systems Architect)
"""

import uuid
import time
import asyncio
import random
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any

from .schemas import (
    FusionRequest,
    FusionResponse,
    ModalityScores,
    ModalityScoreDetail,
    AuditInfo,
    AvailableModalities,
    EnsembleMode,
    ModalityStatus,
    ShapValues,
    Explanation,
)
from .config import (
    MODEL_VERSION,
    USE_MOCK_APIS,
    MOCK_LATENCY,
    CHANNEL_MODALITY_DEFAULTS,
)
from .ensemble import (
    renormalize_weights,
    compute_fraud_score,
    determine_mode,
    determine_decision,
    determine_risk_level,
)
from .confidence import (
    calculate_confidence_adjustment,
    apply_fallback_overrides,
)
from .explainer import generate_explanation


class FusionOrchestrator:
    """Main orchestrator for the Guardian Fusion Layer."""
    
    def __init__(self, use_mocks: bool = USE_MOCK_APIS):
        self.use_mocks = use_mocks
    
    async def evaluate(self, request: FusionRequest) -> FusionResponse:
        """Evaluate a transaction for fraud using the adaptive ensemble."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        
        # Step 1: Resolve available modalities
        modalities = self._resolve_modalities(request)
        
        # Step 2: Call modality APIs
        scores, signals, statuses, failed = await self._call_modalities(request, modalities)
        
        # Step 3: Determine ensemble mode and weights
        nlp_used = scores.get("nlp") is not None
        voice_used = scores.get("voice") is not None
        
        ensemble_mode = determine_mode(True, nlp_used, voice_used)
        weights = renormalize_weights(True, nlp_used, voice_used)
        
        # Step 4: Compute fraud score
        fraud_score = compute_fraud_score(scores, weights)
        
        # Step 5: Calculate confidence adjustment
        confidence_adj = calculate_confidence_adjustment(
            fraud_score, scores, ensemble_mode,
            modalities.nlp, modalities.voice, failed
        )
        
        # Step 6: Determine decision with overrides
        base_decision = determine_decision(fraud_score, request.options.threshold_override)
        final_decision, _ = apply_fallback_overrides(
            fraud_score, confidence_adj.final_confidence, base_decision,
            ensemble_mode, request.options.force_review_on_fallback
        )
        
        # Step 7: Determine risk level
        risk_level = determine_risk_level(fraud_score)
        
        # Step 8: Generate explanations
        if request.options.explain:
            explanation = generate_explanation(
                fraud_score, scores, signals, weights, ensemble_mode,
                modalities.nlp, modalities.voice, failed
            )
        else:
            explanation = Explanation(
                top_factors=[],
                shap_values=ShapValues(numerical_contribution=0.0),
                narrative="Explanation not requested.",
                missing_modality_note=None,
            )
        
        # Step 9: Build modality scores response
        modality_scores_resp = ModalityScores(
            numerical=ModalityScoreDetail(
                score=scores.get("numerical"),
                status=statuses.get("numerical", ModalityStatus.OK),
                weight=weights["numerical"],
                used=True,
            ),
            nlp=ModalityScoreDetail(
                score=scores.get("nlp"),
                status=statuses.get("nlp", ModalityStatus.UNAVAILABLE),
                weight=weights["nlp"],
                used=nlp_used,
            ),
            voice=ModalityScoreDetail(
                score=scores.get("voice"),
                status=statuses.get("voice", ModalityStatus.UNAVAILABLE),
                weight=weights["voice"],
                used=voice_used,
            ),
        )
        
        # Step 10: Build audit info
        processing_time = (time.perf_counter() - start_time) * 1000
        
        modalities_requested = ["numerical"]
        if modalities.nlp:
            modalities_requested.append("nlp")
        if modalities.voice:
            modalities_requested.append("voice")
        
        modalities_used = ["numerical"]
        if nlp_used:
            modalities_used.append("nlp")
        if voice_used:
            modalities_used.append("voice")
        
        audit = AuditInfo(
            request_id=request_id,
            processing_time_ms=round(processing_time, 2),
            modalities_requested=modalities_requested,
            modalities_used=modalities_used,
            modalities_failed=failed,
        )
        
        return FusionResponse(
            transaction_id=request.transaction_id,
            fraud_score=round(fraud_score, 4),
            decision=final_decision,
            confidence=round(confidence_adj.final_confidence, 4),
            risk_level=risk_level,
            ensemble_mode=ensemble_mode,
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow(),
            modality_scores=modality_scores_resp,
            confidence_adjustment=confidence_adj,
            explanation=explanation,
            audit=audit,
        )
    
    def _resolve_modalities(self, request: FusionRequest) -> AvailableModalities:
        """Resolve which modalities to use based on request and channel."""
        if request.available_modalities is not None:
            return AvailableModalities(
                numerical=True,
                nlp=request.available_modalities.nlp,
                voice=request.available_modalities.voice,
            )
        
        channel = request.transaction_data.channel.value
        defaults = CHANNEL_MODALITY_DEFAULTS.get(channel, {
            "numerical": True, "nlp": True, "voice": False
        })
        
        voice_available = defaults["voice"]
        if voice_available and request.voice_payload is None:
            voice_available = False
        elif voice_available and request.voice_payload is not None:
            if not request.voice_payload.audio_url and not request.voice_payload.audio_base64:
                voice_available = False
        
        nlp_available = defaults["nlp"]
        if nlp_available and request.text_payload is None:
            nlp_available = False
        elif nlp_available and request.text_payload is not None:
            if not any([
                request.text_payload.merchant_text,
                request.text_payload.narrative_text,
                request.text_payload.invoice_text,
            ]):
                nlp_available = False
        
        return AvailableModalities(numerical=True, nlp=nlp_available, voice=voice_available)
    
    async def _call_modalities(
        self,
        request: FusionRequest,
        modalities: AvailableModalities,
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, float]], Dict[str, ModalityStatus], List[str]]:
        """Call all requested modality APIs and collect results."""
        scores: Dict[str, Optional[float]] = {}
        signals: Dict[str, Dict[str, float]] = {}
        statuses: Dict[str, ModalityStatus] = {}
        failed: List[str] = []
        
        # Always call Numerical (required)
        try:
            num_score, num_signals = await self._mock_numerical(request)
            scores["numerical"] = num_score
            signals["numerical"] = num_signals
            statuses["numerical"] = ModalityStatus.OK
        except Exception as e:
            raise ValueError(f"Numerical modality failed (required): {e}")
        
        # Call NLP if available
        if modalities.nlp:
            try:
                nlp_score, nlp_signals = await self._mock_nlp(request)
                scores["nlp"] = nlp_score
                signals["nlp"] = nlp_signals
                statuses["nlp"] = ModalityStatus.OK
            except Exception:
                scores["nlp"] = None
                signals["nlp"] = {}
                statuses["nlp"] = ModalityStatus.ERROR
                failed.append("nlp")
        else:
            statuses["nlp"] = ModalityStatus.UNAVAILABLE
        
        # Call Voice if available
        if modalities.voice:
            try:
                voice_score, voice_signals = await self._mock_voice(request)
                scores["voice"] = voice_score
                signals["voice"] = voice_signals
                statuses["voice"] = ModalityStatus.OK
            except Exception:
                scores["voice"] = None
                signals["voice"] = {}
                statuses["voice"] = ModalityStatus.ERROR
                failed.append("voice")
        else:
            statuses["voice"] = ModalityStatus.UNAVAILABLE
        
        return scores, signals, statuses, failed
    
    # =========================================================================
    # MOCK IMPLEMENTATIONS - For prototype demonstration
    # =========================================================================
    
    async def _mock_numerical(self, request: FusionRequest) -> Tuple[float, Dict[str, float]]:
        """Generate realistic mock Numerical stream response."""
        await asyncio.sleep(MOCK_LATENCY["numerical"] / 1000)
        
        amount = request.transaction_data.amount
        
        if amount > 5000:
            amount_anomaly = min(0.95, 0.5 + (amount - 5000) / 20000)
        elif amount < 10:
            amount_anomaly = 0.6
        else:
            amount_anomaly = max(0.1, min(0.4, amount / 10000))
        
        velocity_risk = random.uniform(0.1, 0.5)
        
        category = request.transaction_data.merchant_category or "unknown"
        high_risk_categories = ["gambling", "crypto", "wire_transfer", "cash_advance"]
        if category.lower() in high_risk_categories:
            pattern_deviation = random.uniform(0.6, 0.9)
        else:
            pattern_deviation = random.uniform(0.1, 0.4)
        
        country = request.transaction_data.country or "US"
        high_risk_countries = ["NG", "RU", "CN", "BR", "IN"]
        if country.upper() in high_risk_countries:
            geo_risk = random.uniform(0.5, 0.8)
        else:
            geo_risk = random.uniform(0.1, 0.3)
        
        signals = {
            "amount_anomaly": round(amount_anomaly, 4),
            "velocity_risk": round(velocity_risk, 4),
            "pattern_deviation": round(pattern_deviation, 4),
            "geo_risk": round(geo_risk, 4),
        }
        
        score = (amount_anomaly * 0.35 + velocity_risk * 0.25 +
                 pattern_deviation * 0.25 + geo_risk * 0.15)
        
        return round(score, 4), signals
    
    async def _mock_nlp(self, request: FusionRequest) -> Tuple[float, Dict[str, float]]:
        """Generate realistic mock NLP stream response."""
        await asyncio.sleep(MOCK_LATENCY["nlp"] / 1000)
        
        text_payload = request.text_payload
        merchant_text = text_payload.merchant_text if text_payload else ""
        narrative_text = text_payload.narrative_text if text_payload else ""
        
        typosquatting_risk = 0.1
        suspicious_patterns = ["amaz0n", "paypa1", "app1e", "g00gle", "micr0soft"]
        if merchant_text:
            merchant_lower = merchant_text.lower()
            for pattern in suspicious_patterns:
                if pattern in merchant_lower:
                    typosquatting_risk = random.uniform(0.75, 0.95)
                    break
            if any(char in merchant_lower for char in "01"):
                typosquatting_risk = max(typosquatting_risk, random.uniform(0.5, 0.7))
        
        all_text = f"{merchant_text} {narrative_text}".lower()
        high_risk_words = ["urgent", "wire", "bitcoin", "crypto", "prize", "winner", "lottery"]
        semantic_risk = 0.2
        for word in high_risk_words:
            if word in all_text:
                semantic_risk = max(semantic_risk, random.uniform(0.5, 0.8))
        
        entity_inconsistency = random.uniform(0.1, 0.4)
        
        signals = {
            "semantic_risk": round(semantic_risk, 4),
            "typosquatting_risk": round(typosquatting_risk, 4),
            "entity_inconsistency": round(entity_inconsistency, 4),
        }
        
        score = semantic_risk * 0.35 + typosquatting_risk * 0.45 + entity_inconsistency * 0.20
        
        return round(score, 4), signals
    
    async def _mock_voice(self, request: FusionRequest) -> Tuple[float, Dict[str, float]]:
        """Generate realistic mock Voice stream response."""
        await asyncio.sleep(MOCK_LATENCY["voice"] / 1000)
        
        deepfake_probability = random.uniform(0.05, 0.3)
        voice_stress_indicator = random.uniform(0.1, 0.5)
        speaker_consistency_risk = random.uniform(0.05, 0.25)
        
        signals = {
            "deepfake_probability": round(deepfake_probability, 4),
            "voice_stress_indicator": round(voice_stress_indicator, 4),
            "speaker_consistency": round(1 - speaker_consistency_risk, 4),
        }
        
        score = deepfake_probability * 0.50 + voice_stress_indicator * 0.25 + speaker_consistency_risk * 0.25
        
        return round(score, 4), signals