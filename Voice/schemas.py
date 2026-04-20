from pydantic import BaseModel

class ScoreResponse(BaseModel):
    label: str
    bonafide_prob: float
    spoof_prob: float
    threshold: float
    filename: str


class TranscriptionResponse(BaseModel):
    filename: str
    transcription: dict