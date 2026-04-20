from .config import MODEL_NAME, DEVICE, MODEL_PATH, THRESHOLD_PATH, HF_REPO_ID, MODEL_FILENAME, THRESHOLD_FILENAME
import joblib
from transformers import AutoFeatureExtractor, AutoModel
from fastapi import HTTPException
from pathlib import Path
from huggingface_hub import hf_hub_download
import os

def _download_from_hub(filename: str, target_path: Path):
    """Download a file from Hugging Face Hub if not present locally."""
    if target_path.exists():
        return
    
    print(f"⬇️  Downloading {filename} from {HF_REPO_ID}...")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
        )
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download {filename} from Hub: {e}"
        )

# Ensure model files exist (download if missing)
_download_from_hub(MODEL_FILENAME, MODEL_PATH)
_download_from_hub(THRESHOLD_FILENAME, THRESHOLD_PATH)

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

try:
    THRESHOLD = load_threshold()
except Exception:
    THRESHOLD = None

# Load WavLM
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
wavlm = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
wavlm.eval()