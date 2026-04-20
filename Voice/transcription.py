def transcribe_audio_file(file_path: str, client):
    with open(file_path, "rb") as f:
        return client.speech_to_text.convert(
            file=f,
            model_id="scribe_v2",
            language_code=None,
            diarize=True,
            tag_audio_events=True
        )
    