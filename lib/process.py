import os

from pydub.audio_segment import AudioSegment
from config import default_audio_proc_format
from lib.tts import get_audio

def process_sentence(sentence_data):
    """Process a single sentence chunk and return the result"""
    if len(sentence_data) == 4:  # New format with voice parameter
        sentence_num, sentence, sentence_file, voice = sentence_data
    else:  # Backward compatibility
        sentence_num, sentence, sentence_file = sentence_data
        voice = 'af_heart'

    if os.path.exists(sentence_file):
        # File already exists, skip processing
        return sentence_num, sentence_file, True

    # Handle special pause indicator
    if sentence == "[PAUSE]":
        # Create a short pause (0.5 second of silence)
        silence = AudioSegment.silent(duration=500)  # 500ms
        silence.export(sentence_file, format=default_audio_proc_format)
        return sentence_num, sentence_file, True

    # Get audio from API
    audio_data = get_audio(
        text=sentence, voice=voice
    )

    if audio_data:
        # Save the audio file
        with open(sentence_file, 'wb') as f:
            f.write(audio_data)
        return sentence_num, sentence_file, True
    else:
        return sentence_num, sentence_file, False
