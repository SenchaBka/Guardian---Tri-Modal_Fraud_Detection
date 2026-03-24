import numpy as np
import torch

from model_loader import feature_extractor, wavlm, DEVICE

@torch.no_grad()
def extract_embeddings(audio_1d: np.ndarray) -> np.ndarray:
    """
    This function converts raw audio into a fixed-dimensional
    embedding using a pretrained WavLM model, which is then
    used by a downstream classifier for spoof detection.
    """
    inputs = feature_extractor(audio_1d, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    hidden = wavlm(**inputs).last_hidden_state  # (B,T,D)
    #mean = hidden.mean(dim=1)
    #std = hidden.std(dim=1)
    #emb = torch.cat([mean, std], dim=1).squeeze(0)
    emb = hidden.mean(dim=1).squeeze(0)
    return emb.detach().cpu().numpy().astype(np.float32)