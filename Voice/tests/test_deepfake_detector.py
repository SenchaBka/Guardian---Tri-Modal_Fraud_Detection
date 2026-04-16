import sys
from unittest.mock import patch, MagicMock
import numpy as np

# Mock model loading before importing deepfake_detector
sys.modules["Voice.model_loader"] = MagicMock(
    feature_extractor=MagicMock(),
    wavlm=MagicMock(),
    clf=MagicMock(),
    DEVICE="cpu",
    THRESHOLD=0.5
)

from ..deepfake_detector import score_audio


@patch("Voice.deepfake_detector.clf")
@patch("Voice.deepfake_detector.extract_embeddings")
def test_score_audio_bonafide(mock_extract, mock_clf):
    """Test score_audio classifies high-confidence bonafide audio."""
    mock_extract.return_value = np.array([0.1, 0.2, 0.3, 0.4])
    mock_clf.predict_proba.return_value = np.array([[0.05, 0.95]])
    
    result = score_audio(b"fake audio")
    
    assert result["label"] == "BONAFIDE"
    assert result["bonafide_prob"] == 0.95
    assert result["spoof_prob"] == 0.05
    assert isinstance(result["threshold"], float)


@patch("Voice.deepfake_detector.clf")
@patch("Voice.deepfake_detector.extract_embeddings")
def test_score_audio_spoof(mock_extract, mock_clf):
    """Test score_audio classifies low-confidence audio as spoof."""
    mock_extract.return_value = np.array([0.1, 0.2, 0.3, 0.4])
    mock_clf.predict_proba.return_value = np.array([[0.7, 0.3]])
    
    result = score_audio(b"fake audio")
    
    assert result["label"] == "SPOOF"
    assert result["bonafide_prob"] == 0.3
    assert result["spoof_prob"] == 0.7


@patch("Voice.deepfake_detector.clf")
@patch("Voice.deepfake_detector.extract_embeddings")
def test_score_audio_rounding(mock_extract, mock_clf):
    """Test rounding to 6 decimals and all required keys present."""
    mock_extract.return_value = np.array([0.1, 0.2, 0.3, 0.4])
    mock_clf.predict_proba.return_value = np.array([[0.123456789, 0.876543211]])
    
    result = score_audio(b"fake audio")
    
    # All keys present
    assert set(result.keys()) == {"label", "bonafide_prob", "spoof_prob", "threshold"}
    # Rounded to 6 decimals
    assert result["bonafide_prob"] == 0.876543
    assert result["spoof_prob"] == 0.123457
