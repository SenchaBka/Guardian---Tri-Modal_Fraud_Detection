from .config import MODEL_NAME, DEVICE, MODEL_PATH, THRESHOLD_PATH
import joblib
from transformers import AutoFeatureExtractor, AutoModel
from fastapi import HTTPException

# Load classifier
try:
    clf = joblib.load(MODEL_PATH)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

# Load threshold
def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise HTTPException(status_code=500, detail="Threshold file missing")

    try:
        return float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())
    except:
        raise HTTPException(status_code=500, detail="Invalid threshold file")

THRESHOLD = load_threshold()

# Load WavLM
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
wavlm = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
wavlm.eval()