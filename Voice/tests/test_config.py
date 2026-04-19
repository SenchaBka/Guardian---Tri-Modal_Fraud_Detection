import pytest
from pathlib import Path
import torch
from Voice.config import (
    get_api_key,
    MODEL_NAME,
    DEVICE,
    MODEL_PATH,
    THRESHOLD_PATH,
    MAX_UPLOAD_MB,
)

def test_missing_api_key(monkeypatch):
    """Test that ValueError is raised when ELEVENLABS_API_KEY is missing."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing ELEVENLABS_API_KEY"):
        get_api_key()


def test_get_api_key_success(monkeypatch):
    """Test successful retrieval of API key when it exists."""
    test_key = "test-api-key-12345"
    monkeypatch.setenv("ELEVENLABS_API_KEY", test_key)

    result = get_api_key()
    
    assert result == test_key


def test_config_constants():
    """Test that all config constants are set correctly."""
    # Model configuration
    assert MODEL_NAME == "microsoft/wavlm-large"
    assert isinstance(MODEL_NAME, str)
    
    # Device configuration
    assert DEVICE in ["cuda", "cpu"]
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert DEVICE == expected_device
    
    # File paths
    assert isinstance(MODEL_PATH, Path)
    assert isinstance(THRESHOLD_PATH, Path)
    
    # Upload limit
    assert MAX_UPLOAD_MB == 30
    assert isinstance(MAX_UPLOAD_MB, int)
    assert MAX_UPLOAD_MB > 0

