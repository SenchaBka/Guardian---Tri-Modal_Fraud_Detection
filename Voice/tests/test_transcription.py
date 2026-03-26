import tempfile
from pathlib import Path
from ..transcription import transcribe_audio_file


def test_transcribe_audio_file():
    """Test transcription calls client with correct parameters."""
    class FakeClient:
        def __init__(self):
            self.speech_to_text = self

        def convert(self, **kwargs):
            # Verify parameters
            assert kwargs["model_id"] == "scribe_v2"
            assert kwargs["diarize"] is True
            assert kwargs["tag_audio_events"] is True
            return "Test"

    # Create temp audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(b"fake audio")
        tmp.flush()
        tmp_path = tmp.name

    try:
        result = transcribe_audio_file(tmp_path, FakeClient())
        assert result == "Test"
    finally:
        Path(tmp_path).unlink()


def test_transcribe_audio_file_not_found():
    """Test FileNotFoundError when file doesn't exist."""
    class FakeClient:
        def __init__(self):
            self.speech_to_text = self

    # This should raise FileNotFoundError when trying to open the file
    try:
        transcribe_audio_file("/nonexistent/file.wav", FakeClient())
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass