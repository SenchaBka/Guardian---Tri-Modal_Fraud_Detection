#/api/v1/voice/score

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoFeatureExtractor, AutoModel

import librosa

from fastapi import HTTPException

# -------------------------
# Config
# -------------------------
MODEL_NAME = "microsoft/wavlm-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLF_PATH = Path("data/ASV/cache/wavlm_lr.joblib")
THR_PATH = Path("data/ASV/cache/threshold.txt")

MAX_UPLOAD_MB = 30

# -------------------------
# App
# -------------------------
app = FastAPI(title="Guardian Voice Scoring API", version="1.0")

# -------------------------
# Startup
# -------------------------
# Load threshold
def _load_threshold() -> float:
    if not THR_PATH.exists():
        raise HTTPException(status_code=500, detail="Threshold file missing")

    try:
        return float(THR_PATH.read_text(encoding="utf-8").strip())
    except:
        raise HTTPException(status_code=500, detail="Invalid threshold file")

# Load model
try:
    clf = joblib.load(CLF_PATH)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
wavlm = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
wavlm.eval()

THRESHOLD = _load_threshold()

# Feature extractor
@torch.no_grad()
def wavlm_embedding(audio_1d: np.ndarray) -> np.ndarray:
    """
    This function converts raw audio into a fixed-dimensional
    embedding using a pretrained WavLM model, which is then
    used by a downstream classifier for spoof detection.
    """
    inputs = feature_extractor(audio_1d, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    hidden = wavlm(**inputs).last_hidden_state  # (B,T,D)
    mean = hidden.mean(dim=1)
    std = hidden.std(dim=1)
    emb = torch.cat([mean, std], dim=1).squeeze(0)
    return emb.detach().cpu().numpy().astype(np.float32)

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

# API endpoint
@app.post("/api/v1/voice/score")
async def score_voice(file: UploadFile = File(...)):
    # Read upload into memory
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB} MB).")

    # Save to a temp file so librosa can read it
    suffix = Path(file.filename or "").suffix.lower() or ".bin"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        audio = load_audio_16k_mono(tmp_path)       # (T,)
        emb = wavlm_embedding(audio).reshape(1, -1) # (1, D)

        bonafide_prob = float(clf.predict_proba(emb)[0, 1])
        spoof_prob = 1.0 - bonafide_prob
        label = "BONAFIDE" if bonafide_prob >= THRESHOLD else "SPOOF"

        return {
            "label": label,
            "bonafide_prob": round(bonafide_prob, 6),
            "spoof_prob": round(spoof_prob, 6),
            "threshold": round(float(THRESHOLD), 6),
            "model": MODEL_NAME,
            "filename": file.filename,
        }

    except ValueError as e:
        # for "Empty or unreadable audio." and similar validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to score audio: {e}")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass