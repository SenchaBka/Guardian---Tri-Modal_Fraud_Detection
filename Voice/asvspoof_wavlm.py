from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from transformers import AutoProcessor, AutoModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve


# =========================
# CONFIG (edit these)
# =========================
TRAIN_PROTOCOL = r"data/ASV/train_protocol.txt"
DEV_PROTOCOL   = r"data/ASV/dev_protocol.txt"

TRAIN_AUDIO_DIR = r"data/ASV/train_audio"
DEV_AUDIO_DIR   = r"data/ASV/dev_audio"

CACHE_DIR = r"data/ASV/cache"  # features cached here (recommended)


# Modern + easy model (stronger than wav2vec2 for many spoof tasks)
MODEL_NAME = "microsoft/wavlm-base"


# =========================
# UTILS
# =========================
def read_protocol(protocol_path: str):
    """
    Reads a protocol file and returns list of (file_id, label_str).
    Assumption: label is the LAST token and is 'bonafide' or 'spoof'.
    Works for many ASVspoof protocols.
    """
    items = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = parts[-1].lower()
            if label not in ("bonafide", "spoof"):
                continue  # skip lines that don't match expected format
            # choose a token that looks like an utterance id (contains '_' and digits)
            file_id = None
            for p in parts:
                if "_" in p and any(ch.isdigit() for ch in p):
                    file_id = p
            if file_id is None:
                file_id = parts[0]  # fallback
            items.append((file_id, label))
    return items


def label_to_int(lbl: str) -> int:
    # 1 = real, 0 = fake
    return 1 if lbl == "bonafide" else 0


def find_audio_path(audio_dir: Path, file_id: str):
    """
    ASVspoof commonly uses .flac; some versions use .wav.
    """
    for ext in (".flac", ".wav"):
        p = audio_dir / f"{file_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Audio not found for {file_id} in {audio_dir}")


def load_audio_mono_16k(path: str) -> torch.Tensor:
    """
    Returns 1D float tensor at 16kHz (mono).
    """
    wav, sr = torchaudio.load(path)  # (channels, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)  # (T,)


def compute_eer(y_true, scores_real_prob):
    """
    y_true: 1=bonafide, 0=spoof
    scores_real_prob: higher means more likely bonafide
    """
    fpr, tpr, thr = roc_curve(y_true, scores_real_prob, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])


# =========================
# FEATURE EXTRACTOR (WavLM)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


@torch.no_grad()
def wavlm_embedding(audio_1d: torch.Tensor) -> np.ndarray:
    """
    Extracts one embedding per utterance using mean pooling over time.
    audio_1d: (T,) float tensor @16kHz
    returns: (hidden_dim,) numpy
    """
    inputs = processor(audio_1d, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    hidden = model(**inputs).last_hidden_state  # (1, frames, dim)
    emb = hidden.mean(dim=1).squeeze(0)         # (dim,)
    return emb.detach().cpu().numpy()


def build_features(items, audio_dir: str, cache_path: str):
    """
    Builds (or loads) X, y from a protocol list.
    Caches to .npz so you don't recompute embeddings every run.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"]

    audio_dir = Path(audio_dir)
    X_list, y_list = [], []

    for file_id, lbl in tqdm(items, desc=f"Embedding {audio_dir.name}"):
        p = find_audio_path(audio_dir, file_id)
        audio = load_audio_mono_16k(str(p))
        emb = wavlm_embedding(audio)

        X_list.append(emb)
        y_list.append(label_to_int(lbl))

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.savez(cache_path, X=X, y=y)
    return X, y


# =========================
# MAIN
# =========================
def main():
    train_items = read_protocol(TRAIN_PROTOCOL)
    dev_items   = read_protocol(DEV_PROTOCOL)

    if len(train_items) == 0 or len(dev_items) == 0:
        raise RuntimeError(
            "Protocol parsing returned 0 items. "
            "Open your protocol file and confirm the last column is bonafide/spoof."
        )

    train_cache = Path(CACHE_DIR) / "train_wavlm.npz"
    dev_cache   = Path(CACHE_DIR) / "dev_wavlm.npz"

    X_train, y_train = build_features(train_items, TRAIN_AUDIO_DIR, str(train_cache))
    X_dev,   y_dev   = build_features(dev_items,   DEV_AUDIO_DIR,   str(dev_cache))

    # Simple strong baseline
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)

    # probability of bonafide (real)
    scores = clf.predict_proba(X_dev)[:, 1]

    eer, thr = compute_eer(y_dev, scores)
    print(f"\nEER = {eer*100:.2f}%  (threshold={thr:.4f})")
    print("Note: lower EER is better.\n")


if __name__ == "__main__":
    main()