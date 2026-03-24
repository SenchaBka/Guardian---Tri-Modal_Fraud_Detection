from pathlib import Path
import torch

MODEL_NAME = "microsoft/wavlm-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLF_PATH = Path("data/ASV/cache/wavlm_lr.joblib")
THR_PATH = Path("data/ASV/cache/threshold.txt")

MAX_UPLOAD_MB = 30