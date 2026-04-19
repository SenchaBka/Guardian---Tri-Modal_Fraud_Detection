from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from ..preprocessor import load_audio_16k_mono


@patch("Voice.preprocessor.librosa.effects.trim")
@patch("Voice.preprocessor.librosa.load")
def test_load_audio_success(mock_load, mock_trim):
    """Test successful audio loading with trim and float32 conversion."""
    raw_audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_load.return_value = (raw_audio, 16000)
    
    trimmed_audio = np.array([0.2, 0.3, 0.4])
    mock_trim.return_value = (trimmed_audio, None)
    
    result = load_audio_16k_mono("test.wav")
    
    # Verify correct parameters
    mock_load.assert_called_once_with("test.wav", sr=16000, mono=True)
    mock_trim.assert_called_once()
    assert mock_trim.call_args[1]["top_db"] == 30
    
    # Verify output
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


@patch("Voice.preprocessor.librosa.load")
def test_load_audio_empty_raises_error(mock_load):
    """Test that empty audio raises ValueError."""
    mock_load.return_value = (np.array([]), 16000)
    
    with pytest.raises(ValueError, match="Empty or unreadable audio"):
        load_audio_16k_mono("empty.wav")
