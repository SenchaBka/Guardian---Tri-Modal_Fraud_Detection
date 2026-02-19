"""
Guardian Fusion Layer - Configuration Constants
================================================

Author: Sherwayne (ML Systems Architect)
"""

from typing import Dict, Tuple

# =============================================================================
# ENSEMBLE WEIGHTS
# =============================================================================

BASE_WEIGHTS: Dict[str, float] = {
    "numerical": 0.45,
    "nlp": 0.35,
    "voice": 0.20,
}

PRECALCULATED_WEIGHTS: Dict[Tuple[bool, bool, bool], Dict[str, float]] = {
    (True, True, True): {"numerical": 0.45, "nlp": 0.35, "voice": 0.20},
    (True, True, False): {"numerical": 0.5625, "nlp": 0.4375, "voice": 0.0},
    (True, False, True): {"numerical": 0.6923, "nlp": 0.0, "voice": 0.3077},
    (True, False, False): {"numerical": 1.0, "nlp": 0.0, "voice": 0.0},
}

# =============================================================================
# CONFIDENCE PENALTIES
# =============================================================================

CONFIDENCE_PENALTIES: Dict[str, float] = {
    "full": 0.00,
    "partial_no_voice": 0.15,
    "partial_no_nlp": 0.20,
    "fallback": 0.40,
}

ERROR_PENALTY_PER_MODALITY: float = 0.05

# =============================================================================
# DECISION THRESHOLDS
# =============================================================================

DECISION_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "approve": (0.00, 0.30),
    "review": (0.31, 0.70),
    "block": (0.71, 1.00),
}

RISK_LEVEL_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "low": (0.00, 0.25),
    "medium": (0.26, 0.50),
    "high": (0.51, 0.75),
    "critical": (0.76, 1.00),
}

# =============================================================================
# FALLBACK OVERRIDE RULES
# =============================================================================

FALLBACK_OVERRIDES = {
    "force_review_threshold": 0.25,
    "force_block_threshold": 0.60,
}

PARTIAL_MODE_OVERRIDES = {
    "low_confidence_threshold": 0.70,
    "force_review_score": 0.40,
}

# =============================================================================
# CHANNEL-BASED AUTO-DETECTION
# =============================================================================

CHANNEL_MODALITY_DEFAULTS: Dict[str, Dict[str, bool]] = {
    "online": {"numerical": True, "nlp": True, "voice": False},
    "phone": {"numerical": True, "nlp": True, "voice": True},
    "in_person": {"numerical": True, "nlp": True, "voice": True},
    "atm": {"numerical": True, "nlp": False, "voice": False},
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

MODEL_VERSION: str = "1.0.0-prototype"

API_TIMEOUTS: Dict[str, float] = {
    "numerical": 2.0,
    "nlp": 3.0,
    "voice": 5.0,
}

MODALITY_ENDPOINTS: Dict[str, str] = {
    "numerical": "http://localhost:8001/api/v1/numerical/score",
    "nlp": "http://localhost:8002/api/v1/nlp/score",
    "voice": "http://localhost:8003/api/v1/voice/score",
}

# =============================================================================
# SHAP & MOCK CONFIGURATION
# =============================================================================

TOP_FACTORS_COUNT: int = 5
SHAP_BASELINE_RATE: float = 0.02
USE_MOCK_APIS: bool = True

MOCK_LATENCY: Dict[str, int] = {
    "numerical": 50,
    "nlp": 75,
    "voice": 100,
}