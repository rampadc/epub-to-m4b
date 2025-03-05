import io
from kokoro_onnx import Kokoro
import soundfile as sf

def get_audio(text):
    """Get audio using Kokoro TTS

    Args:
        text (str): Text to convert to speech

    Returns:
        bytes: Audio content as WAV bytes or None if generation failed
    """
    try:
        # Initialize Kokoro
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

        # Generate audio
        samples, sample_rate = kokoro.create(text, "af_heart", is_phonemes=False)

        if len(samples) == 0:
            print("Error in generating audio")
            return None

        # Convert numpy array to WAV bytes
        with io.BytesIO() as wav_io:
            sf.write(
                wav_io,
                samples,
                samplerate=sample_rate,
                format='WAV',
                subtype='PCM_16'
            )
            return wav_io.getvalue()

    except Exception as e:
        print(f"Kokoro TTS error: {e}")
        return None
