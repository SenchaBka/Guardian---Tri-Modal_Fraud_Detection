from pathlib import Path
import torch

MODEL_NAME = "microsoft/wavlm-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("data/ASV/cache/wavlm_lr.joblib")
THRESHOLD_PATH = Path("data/ASV/cache/threshold.txt")

MAX_UPLOAD_MB = 30