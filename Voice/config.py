from pathlib import Path
import torch
from dotenv import load_dotenv
import os

load_dotenv()

def get_api_key():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("Missing ELEVENLABS_API_KEY")
    return api_key

MODEL_NAME = "microsoft/wavlm-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("data/ASV/cache/wavlm_lr.joblib")
THRESHOLD_PATH = Path("data/ASV/cache/threshold.txt")

MAX_UPLOAD_MB = 30