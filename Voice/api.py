from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from preprocessor import load_audio_16k_mono
from deepfake_detector import score_audio
from transcription import transcribe_audio_file
from config import MAX_UPLOAD_MB

from elevenlabs.client import ElevenLabs
from config import ELEVENLABS_API_KEY

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def get_client():
    return client

app = FastAPI(title="Guardian Voice Scoring API", version="1.1")

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

        audio = load_audio_16k_mono(tmp_path)      
        result = score_audio(audio)             

        return {
            **result,
            "filename": file.filename
        }

    except ValueError as e:
        # for "Empty or unreadable audio." and similar validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to score audio: {e}")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass 


# API endpoint
@app.post("/api/v1/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...), client = Depends(get_client)):
    # Read file
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    # Save temp file
    suffix = Path(file.filename or "").suffix.lower() or ".bin"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        transcription = transcribe_audio_file(tmp_path, client)

        return {
            "filename": file.filename,
            "transcription": transcription
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass