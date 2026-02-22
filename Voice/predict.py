import torch
import torchaudio
from transformers import AutoModel, AutoFeatureExtractor
import joblib
import librosa
from pathlib import Path

# PATHS
AUDIO_PATH = "data/sample_real2.flac"  
MODEL_PATH = "data/ASV/cache/wavlm_lr.joblib"
THRESHOLD_PATH = "data/ASV/cache/threshold.txt"

THRESHOLD = float(Path(THRESHOLD_PATH).read_text().strip())

MODEL_NAME = "microsoft/wavlm-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD CLASSIFIER 
clf = joblib.load(MODEL_PATH)

# LOAD WavLM
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# LOAD AUDIO (MP3/WAV/FLAC) via soundfile
wav, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)

# mono
if wav.ndim == 2:
    wav = wav.mean(axis=1)

wav = torch.from_numpy(wav)

# resample
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)

# EMBEDDING
with torch.no_grad():
    inputs = feature_extractor(wav.numpy(), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    hidden = model(**inputs).last_hidden_state
    emb = hidden.mean(dim=1).cpu().numpy()  # shape (1, D)

# PREDICT 
prob = clf.predict_proba(emb)[0, 1]  # bonafide prob

label = "BONAFIDE (real)" if prob >= THRESHOLD else "SPOOF (fake)"
print(label)
print(f"bonafide_prob={prob:.4f}  threshold={THRESHOLD:.4f}")