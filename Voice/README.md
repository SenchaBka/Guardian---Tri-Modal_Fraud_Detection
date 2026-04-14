To start:
cd Voice
uvicorn api:app --reload

API Endpoints:
1. /api/v1/voice/score
Returns spoof/bonafide probabilities.
Response body example:
{
  "label": "SPOOF",
  "bonafide_prob": 0.000004,
  "spoof_prob": 0.999996,
  "threshold": 0.162777,
  "filename": "sample_fake.flac"
}

2. api/v1/voice/transcribe
Convert speech to text with ElevenLabs API
Response body example:
{
  "filename": "sample_fake.flac",
  "transcription": {
    "language_code": "eng",
    "language_probability": 0.9820188879966736,
    "text": "It was highly successful",
    "words": [
      {
        "text": "It",
        "start": 0.079,
        "end": 0.179,
        "type": "word",
        "speaker_id": "speaker_0",
        "logprob": -0.0007098776986822486
      },
      {
        "text": " ",
        "start": 0.179,
        "end": 0.219,
        "type": "spacing",
        "speaker_id": "speaker_0",
        "logprob": 0
      },
      {
        "text": "was",
        "start": 0.219,
        "end": 0.34,
        "type": "word",
        "speaker_id": "speaker_0",
        "logprob": 0
      },
      {
        "text": " ",
        "start": 0.34,
        "end": 0.379,
        "type": "spacing",
        "speaker_id": "speaker_0",
        "logprob": -9.536738616588991e-7
      },
      {
        "text": "highly",
        "start": 0.379,
        "end": 0.699,
        "type": "word",
        "speaker_id": "speaker_0",
        "logprob": -9.536738616588991e-7
      },
      {
        "text": " ",
        "start": 0.699,
        "end": 0.74,
        "type": "spacing",
        "speaker_id": "speaker_0",
        "logprob": -0.0000017881377516459906
      },
      {
        "text": "successful",
        "start": 0.74,
        "end": 1.379,
        "type": "word",
        "speaker_id": "speaker_0",
        "logprob": -0.0000017881377516459906
      }
    ],
    "transcription_id": "K9lkLcU906yoSsRLdYlC",
    "audio_duration_secs": 1.468
  }
}

AFIs:
1. Changes dataset to more modern one
2. Add logging
3. Improve exception handling
4. Prepare for scaling

