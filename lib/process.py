import os
from pydub.audio_segment import AudioSegment
from config import default_audio_proc_format
from lib.tts import get_audio
import logging

log = logging.getLogger(__name__)

def process_sentence(sentence_data):
    """Process a single sentence and save the audio file."""
    sentence_num, sentence, sentence_file, voice = sentence_data['sentence_num'], sentence_data['text'], sentence_data['file'], sentence_data['voice']


    if sentence_file.exists():
        log.debug(f"Sentence file already exists: {sentence_file}")
        return sentence_num, sentence_file, True  # Return file path

    if sentence == "[PAUSE]":
        silence = AudioSegment.silent(duration=500)
        silence.export(sentence_file, format=default_audio_proc_format)
        log.info(f"Created pause at: {sentence_file}")
        return sentence_num, sentence_file, True

    audio_data = get_audio(text=sentence, voice=voice)
    if audio_data:
        with open(sentence_file, 'wb') as f:
            f.write(audio_data)
        # log.info(f"Processed sentence {sentence_num}: {sentence_file}")
        return sentence_num, sentence_file, True  # Return file path
    else:
        log.error(f"Failed to process sentence {sentence_num}.")
        return sentence_num, sentence_file, False
