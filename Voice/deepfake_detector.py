from .feature_extractor import extract_embeddings
from .model_loader import clf, THRESHOLD

def score_audio(audio):
    emb = extract_embeddings(audio).reshape(1, -1)

    bonafide_prob = float(clf.predict_proba(emb)[0, 1])
    spoof_prob = 1.0 - bonafide_prob
    label = "BONAFIDE" if bonafide_prob >= THRESHOLD else "SPOOF"

    return {
        "label": label,
        "bonafide_prob": round(bonafide_prob, 6),
        "spoof_prob": round(spoof_prob, 6),
        "threshold": round(float(THRESHOLD), 6),
    }