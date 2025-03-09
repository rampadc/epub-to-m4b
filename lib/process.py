from lib.tts import get_audio
import logging
import numpy as np
import wave
from pydub import AudioSegment

log = logging.getLogger(__name__)

def process_sentence(sentence_data):
    """Process a single sentence, handling [PAUSE] by splitting into parts."""
    sentence_num = sentence_data['sentence_num']
    sentence = sentence_data['text']
    wav_file = sentence_data['wav_file']
    voice = sentence_data['voice']

    if wav_file.exists():
        return sentence_num, wav_file, True

    if sentence == "[PAUSE]":
        silence = AudioSegment.silent(duration=500)
        silence.export(wav_file, format="wav")
        return sentence_num, wav_file, True

    parts = sentence.split("[PAUSE]")
    all_samples = []
    sample_rate = None  # Initialize sample_rate

    for part in parts:
        part = part.strip()
        if not part:  # Skip empty parts
            continue

        samples, current_sample_rate = get_audio(text=part, voice=voice)

        if samples is not None and current_sample_rate is not None:
            if sample_rate is None:
                sample_rate = current_sample_rate  # Use the first part's sample rate
            elif sample_rate != current_sample_rate:
                log.error(f"Sample rate mismatch in sentence {sentence_num} - part: '{part}'")
                return sentence_num, wav_file, False  # Or handle differently

            all_samples.append(samples)

            # Add 500ms of silence if there's a pause *after* this part and it isn't last.
            if "[PAUSE]" in sentence and part != parts[-1]:
                silence_samples = np.zeros(int(0.5 * sample_rate), dtype=np.float32)  # 500ms silence
                all_samples.append(silence_samples)

        else:  # Handle get_audio failure for a part
            log.error(f"Failed to generate audio for part '{part}' of sentence {sentence_num}")
            #  Could return False here, or attempt a fallback.  Returning False is simpler.
            return sentence_num, wav_file, False

    if not all_samples:
        log.error(f"No audio generated for sentence {sentence_num}")
        return sentence_num, wav_file, False

    # Combine samples
    combined_samples = np.concatenate(all_samples)

    try:
        # Convert to 16-bit int.
        combined_samples = (combined_samples * 32767).astype(np.int16)
        wav_file.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(combined_samples.tobytes())

        return sentence_num, wav_file, True

    except Exception as e:
        log.error(f"Failed to create WAV for sentence {sentence_num}: {e}")
        return sentence_num, wav_file, False
