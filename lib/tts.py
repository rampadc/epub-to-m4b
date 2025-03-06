import io
from kokoro_onnx import Kokoro
import soundfile as sf
from onnxruntime import InferenceSession
import onnxruntime
import os
cpu_count = os.cpu_count()

sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = os.cpu_count()

session = InferenceSession(
    "kokoro-v1.0.onnx", providers=['CoreMLExecutionProvider'], sess_options=sess_options
)
kokoro = Kokoro.from_session(session, "voices-v1.0.bin")

def get_voices():
    return kokoro.get_voices()

def get_audio(text, voice):
    """Get audio using Kokoro TTS

    Args:
        text (str): Text to convert to speech

    Returns:
        bytes: Audio content as WAV bytes or None if generation failed
    """
    try:
        # Generate audio
        samples, sample_rate = kokoro.create(text, voice, is_phonemes=False)

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
