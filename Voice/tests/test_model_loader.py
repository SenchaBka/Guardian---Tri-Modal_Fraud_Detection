import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

# Remove any MagicMock injected by other test files
sys.modules.pop("Voice.model_loader", None)

# Mock heavy dependencies before importing the real model_loader
with patch("joblib.load") as mock_joblib, \
     patch("transformers.AutoFeatureExtractor.from_pretrained") as mock_feat_ext, \
     patch("transformers.AutoModel.from_pretrained") as mock_model, \
     patch("Voice.config.THRESHOLD_PATH") as mock_threshold_path:
    
    mock_model.return_value = MagicMock()
    mock_feat_ext.return_value = MagicMock()
    mock_threshold_path.exists.return_value = True
    mock_threshold_path.read_text.return_value = "0.5"
    
    import importlib
    import Voice.model_loader
    importlib.reload(Voice.model_loader)
    from Voice.model_loader import load_threshold


@patch("Voice.model_loader.THRESHOLD_PATH")
def test_load_threshold_success(mock_threshold_path):
    """Test successful threshold loading from file."""
    mock_threshold_path.exists.return_value = True
    mock_threshold_path.read_text.return_value = "0.5\n"
    
    result = load_threshold()
    
    assert result == 0.5
    assert isinstance(result, float)


@patch("Voice.model_loader.THRESHOLD_PATH")
def test_load_threshold_file_missing(mock_threshold_path):
    """Test that HTTPException is raised when threshold file doesn't exist."""
    mock_threshold_path.exists.return_value = False
    
    with pytest.raises(Exception, match="Threshold file missing"):
        load_threshold()


@patch("Voice.model_loader.THRESHOLD_PATH")
def test_load_threshold_invalid_content(mock_threshold_path):
    """Test that HTTPException is raised when file content is not a valid float."""
    mock_threshold_path.exists.return_value = True
    mock_threshold_path.read_text.return_value = "invalid_number"
    
    with pytest.raises(Exception, match="Invalid threshold file"):
        load_threshold()


@patch("Voice.model_loader.THRESHOLD_PATH")
def test_load_threshold_strips_whitespace(mock_threshold_path):
    """Test that threshold loading strips whitespace."""
    mock_threshold_path.exists.return_value = True
    mock_threshold_path.read_text.return_value = "  0.75  \n"
    
    result = load_threshold()
    
    assert result == 0.75
