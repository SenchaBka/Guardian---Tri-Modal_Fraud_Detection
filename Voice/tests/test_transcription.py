from ..transcription import transcribe_audio_file

def test_transcribe_audio_file():
    # This test assumes you have a valid audio file
    assert transcribe_audio_file("../data/sample_real2.flac") is not None

    