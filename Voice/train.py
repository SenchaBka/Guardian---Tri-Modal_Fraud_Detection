# =========================
# SETUP
# =========================
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from transformers import AutoModel, AutoFeatureExtractor

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

import soundfile as sf
import joblib


TRAIN_PROTOCOL = r"data/ASV/ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL   = r"data/ASV/ASVspoof2019.LA.cm.dev.trl.txt"

TRAIN_AUDIO_DIR = r"data/ASV/train_audio/ASVspoof2019_LA_train/flac"
DEV_AUDIO_DIR   = r"data/ASV/dev_audio/ASVspoof2019_LA_dev/flac"

CACHE_DIR = r"data/ASV/cache"  # features cached here

# Modern + easy model (stronger than wav2vec2 for many spoof tasks)
MODEL_NAME = "microsoft/wavlm-base"


# =========================
# UTILITIES
# =========================
def index_audio_files(audio_dir: str):
    '''
    Builds a fast lookup set (sets are faster than lists)
    of all audio file names (without extensions) in a folder.
    '''
    audio_dir = Path(audio_dir) 
    stems = set(p.stem for p in audio_dir.glob("*.flac"))
    return stems

def read_protocol(protocol_path: str, audio_dir: str):
    """
    Reads an ASVspoof protocol file and returns a list of (file_id, label).

    The function:
    1) Loads all existing audio filenames from the dataset folder
    2) Parses each line of the protocol file
    3) Extracts the correct file_id by matching against actual files
    4) Extracts the label (bonafide / spoof)

    Returns:
        List of tuples: [(file_id, label), ...]
    """
    existing = index_audio_files(audio_dir)

    items = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            label = parts[-1].lower()
            if label not in ("bonafide", "spoof"):
                continue

            file_id = None
            for tok in parts:
                if tok in existing:
                    file_id = tok
                    break

            if file_id is None:
                # fallback: use the 2nd column (often the audio id in CM protocols)
                file_id = parts[1]

            items.append((file_id, label))

    return items


def label_to_int(lbl: str) -> int:
    '''
    Converts labels to 0 or 1
    '''
    # 1 = real, 0 = fake
    return 1 if lbl == "bonafide" else 0


def find_audio_path(audio_dir: Path, file_id: str):
    '''
    Finds if audio file exists for each label from protocol
    '''
    p = Path(audio_dir) / f"{file_id}.flac"
    if p.exists():
        return p
    raise FileNotFoundError(f"Audio not found for {file_id} in {audio_dir}")


def load_audio_mono_16k(path: str) -> torch.Tensor:
    '''
    Standardizes audio so every file is:
        same channel count (mono)
        same sample rate (16k)
        compatible with WavLM feature extractor
    '''
    wav, sr = sf.read(path, dtype="float32")  # (T,) or (T, C)

    # Convert to mono (if it’s stereo)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav_t = torch.from_numpy(wav) # converts a NumPy array into a PyTorch tensor

    # Resample only if needed
    if sr != 16000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)

    return wav_t


def compute_eer(y_true, scores_real_prob):
    """
    Computes EER (Equal Error Rate)
    The standard metric for spoof detection

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
'''
This block sets up a pretrained audio model (WavLM)
ready to convert raw audio into embeddings efficiently
and correctly.
'''
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


@torch.no_grad() # Turns off gradient computation because not training, just extracting features
def wavlm_embedding(audio_1d: torch.Tensor) -> np.ndarray:
    '''
    Turns raw audio into something the classifier can use.
    '''
    # Converts raw waveform to model-ready format
    inputs = feature_extractor(
        audio_1d.numpy(),  # feature_extractor expects numpy/list
        sampling_rate=16000,
        return_tensors="pt"
    )
    # Moves all tensors to same device as model
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Forward pass through model
    hidden = model(**inputs).last_hidden_state

    # Summarize the whole audio into one vector
    emb = hidden.mean(dim=1).squeeze(0)

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
    train_items = read_protocol(TRAIN_PROTOCOL, TRAIN_AUDIO_DIR)
    dev_items   = read_protocol(DEV_PROTOCOL,   DEV_AUDIO_DIR)

    if len(train_items) == 0 or len(dev_items) == 0:
        raise RuntimeError(
            "Protocol parsing returned 0 items. "
            "Open your protocol file and confirm the last column is bonafide/spoof."
        )

    # Define where embeddings will be saved
    train_cache = Path(CACHE_DIR) / "train_wavlm.npz"
    dev_cache   = Path(CACHE_DIR) / "dev_wavlm.npz"

    # Build features
    X_train, y_train = build_features(train_items, TRAIN_AUDIO_DIR, str(train_cache))
    X_dev,   y_dev   = build_features(dev_items,   DEV_AUDIO_DIR,   str(dev_cache))

    # Train the model
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # Probability of bonafide (real)
    scores = clf.predict_proba(X_dev)[:, 1]

    # Compute and display EER
    eer, thr = compute_eer(y_dev, scores)
    print(f"\nEER = {eer*100:.2f}%  (threshold={thr:.4f})")

    # Save the model
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, Path(CACHE_DIR) / "wavlm_lr.joblib")

    # Save threshold
    (Path(CACHE_DIR) / "threshold.txt").write_text(str(thr))


if __name__ == "__main__":
    main()