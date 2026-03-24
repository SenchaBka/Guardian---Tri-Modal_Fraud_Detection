from config import MODEL_NAME, DEVICE, CLF_PATH, THR_PATH
import joblib
from transformers import AutoFeatureExtractor, AutoModel
from fastapi import HTTPException

# Load classifier
try:
    clf = joblib.load(CLF_PATH)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

# Load threshold
def load_threshold() -> float:
    if not THR_PATH.exists():
        raise HTTPException(status_code=500, detail="Threshold file missing")

    try:
        return float(THR_PATH.read_text(encoding="utf-8").strip())
    except:
        raise HTTPException(status_code=500, detail="Invalid threshold file")

THRESHOLD = load_threshold()

# Load WavLM
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
wavlm = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
wavlm.eval()