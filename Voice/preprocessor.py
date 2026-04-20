import librosa
import numpy as np

# Audio loader
def load_audio_16k_mono(path: str) -> np.ndarray:
    """
    Loads many formats (mp3/m4a/wav/flac) and returns
    16kHz mono float32 waveform.
    """
    y, _ = librosa.load(path, sr=16000, mono=True)

    # y = y[:16000 * 5]  # max 5 seconds

    if y is None or len(y) == 0:
        raise ValueError("Empty or unreadable audio.")

    # remove silence
    y, _ = librosa.effects.trim(y, top_db=30)

    return y.astype(np.float32)