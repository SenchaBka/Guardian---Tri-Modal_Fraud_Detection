import sys
from io import BytesIO
from unittest.mock import patch, MagicMock

# Mock model loading before importing app
sys.modules["Voice.model_loader"] = MagicMock(
    feature_extractor=MagicMock(),
    wavlm=MagicMock(),
    clf=MagicMock(),
    DEVICE="cpu",
    THRESHOLD=0.5
)

from fastapi.testclient import TestClient
from ..api import app


client = TestClient(app)


def test_score_voice_empty_file():
    """Test score endpoint with empty file."""
    response = client.post(
        "/api/v1/voice/score",
        files={"file": ("test.wav", BytesIO(b""), "audio/wav")}
    )
    assert response.status_code == 400
    assert "Empty upload" in response.json()["detail"]


def test_score_voice_file_too_large():
    """Test score endpoint with file larger than MAX_UPLOAD_MB."""
    large_data = b"x" * (31 * 1024 * 1024)  # 31 MB (limit is 30 MB)
    response = client.post(
        "/api/v1/voice/score",
        files={"file": ("test.wav", BytesIO(large_data), "audio/wav")}
    )
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


@patch("Voice.api.score_audio")
@patch("Voice.api.load_audio_16k_mono")
def test_score_voice_success(mock_load_audio, mock_score):
    """Test successful score endpoint."""
    # Mock the audio loading and scoring
    mock_load_audio.return_value = [[0.1, 0.2, 0.3]]
    mock_score.return_value = {
        "label": "BONAFIDE",
        "bonafide_prob": 0.95,
        "spoof_prob": 0.05,
        "threshold": 0.5
    }

    response = client.post(
        "/api/v1/voice/score",
        files={"file": ("test.wav", BytesIO(b"fake audio"), "audio/wav")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "BONAFIDE"
    assert data["filename"] == "test.wav"


def test_transcribe_audio_empty_file():
    """Test transcribe endpoint with empty file."""
    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("test.wav", BytesIO(b""), "audio/wav")}
    )
    assert response.status_code == 400
    assert "Empty upload" in response.json()["detail"]
