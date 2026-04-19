import sys
from unittest.mock import patch, MagicMock
import numpy as np
import torch

# Mock model loading before importing feature_extractor
sys.modules["Voice.model_loader"] = MagicMock(
    feature_extractor=MagicMock(),
    wavlm=MagicMock(),
    DEVICE="cpu"
)

from ..feature_extractor import extract_embeddings


@patch("Voice.feature_extractor.wavlm")
@patch("Voice.feature_extractor.feature_extractor")
def test_extract_embeddings_returns_numpy_array(mock_feat_ext, mock_wavlm):
    """Test that extract_embeddings returns numpy array of correct type."""
    # Mock feature_extractor output
    mock_feat_ext.return_value = {"input_values": torch.randn(1, 16000)}
    
    # Mock wavlm output
    mock_hidden = torch.randn(1, 100, 768)  # (batch, time, dim)
    mock_wavlm.return_value = MagicMock(last_hidden_state=mock_hidden)
    
    result = extract_embeddings(np.random.randn(16000).astype(np.float32))
    
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


@patch("Voice.feature_extractor.wavlm")
@patch("Voice.feature_extractor.feature_extractor")
def test_extract_embeddings_correct_shape(mock_feat_ext, mock_wavlm):
    """Test that extract_embeddings output has correct shape (1D)."""
    mock_feat_ext.return_value = {"input_values": torch.randn(1, 16000)}
    
    # Hidden state: (batch=1, time=100, dim=768)
    mock_hidden = torch.randn(1, 100, 768)
    mock_wavlm.return_value = MagicMock(last_hidden_state=mock_hidden)
    
    result = extract_embeddings(np.random.randn(16000).astype(np.float32))
    
    # After mean(dim=1): (1, 768), then squeeze(0): (768,)
    assert result.shape == (768,)


@patch("Voice.feature_extractor.wavlm")
@patch("Voice.feature_extractor.feature_extractor")
def test_extract_embeddings_calls_feature_extractor_with_16k_sr(mock_feat_ext, mock_wavlm):
    """Test that feature_extractor is called with 16000 sampling rate."""
    mock_feat_ext.return_value = {"input_values": torch.randn(1, 16000)}
    mock_hidden = torch.randn(1, 100, 768)
    mock_wavlm.return_value = MagicMock(last_hidden_state=mock_hidden)
    
    audio = np.random.randn(16000).astype(np.float32)
    extract_embeddings(audio)
    
    # Verify feature_extractor was called with audio and sampling_rate=16000
    mock_feat_ext.assert_called_once()
    call_args = mock_feat_ext.call_args
    assert call_args[1]["sampling_rate"] == 16000
    assert call_args[1]["return_tensors"] == "pt"
