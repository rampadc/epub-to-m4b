from pydub import AudioSegment
from lib.tts import get_audio
import logging

log = logging.getLogger(__name__)

def process_sentence(sentence_data):
    """Process a single sentence and save as WAV for intermediate processing."""
    import numpy as np
    import wave

    sentence_num = sentence_data['sentence_num']
    sentence = sentence_data['text']
    mp3_file = sentence_data['file']  # Keep track of final MP3 path
    wav_file = sentence_data['wav_file']  # Temporary WAV path
    voice = sentence_data['voice']

    # If MP3 exists, we've already processed this sentence
    if mp3_file.exists():
        return sentence_num, mp3_file, wav_file, True

    if sentence == "[PAUSE]":
        silence = AudioSegment.silent(duration=500)
        silence.export(wav_file, format="wav")
        return sentence_num, mp3_file, wav_file, True

    samples, sample_rate = get_audio(text=sentence, voice=voice)

    if samples is not None and sample_rate is not None:
        try:
            # Convert to 16-bit int
            samples = (samples * 32767).astype(np.int16)

            # Create WAV file directory if it doesn't exist
            wav_file.parent.mkdir(parents=True, exist_ok=True)

            # Write WAV file directly
            with wave.open(str(wav_file), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())

            return sentence_num, mp3_file, wav_file, True

        except Exception as e:
            log.error(f"Failed to create WAV for sentence {sentence_num}: {e}")
            return sentence_num, mp3_file, wav_file, False
    else:
        log.error(f"Failed to generate audio for sentence {sentence_num}")
        return sentence_num, mp3_file, wav_file, False
