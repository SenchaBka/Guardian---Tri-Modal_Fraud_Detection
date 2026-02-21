"""
Guardian Numerical Stream
-------------------------
Transactional intelligence layer of the Guardian Tri-Modal Fraud Detection System.

Exposes a calibrated fraud probability (score_numerical ∈ [0, 1]) and four
interpretable sub-signals consumed by the Fusion/Decision Layer.

Owner  : Ivan – ML Engineer (Numerical)
Version: 1.0.0
"""

from inference import score_transaction
from schemas import NumericalScoreRequest, NumericalScoreResponse, StreamStatus

__all__ = [
    "score_transaction",
    "NumericalScoreRequest",
    "NumericalScoreResponse",
    "StreamStatus",
]
