import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv  

load_dotenv() 

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def transcribe_audio_file(file_path: str):
    with open(file_path, "rb") as f:
        transcription = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v2",
            language_code=None,
            diarize=True,
            tag_audio_events=True
        )
    return transcription