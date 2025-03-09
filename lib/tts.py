import io
import logging
from kokoro_onnx import Kokoro
import soundfile as sf
from onnxruntime import InferenceSession, get_available_providers
import onnxruntime
import os
import re

log = logging.getLogger(__name__)

# Configure the onnxruntime session
cpu_count = os.cpu_count()
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = cpu_count

# Set up logging for phonemizer to reduce warning noise
phonemizer_logger = logging.getLogger('phonemizer')
phonemizer_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

def get_best_execution_provider():
    """
    Gets the best available execution provider, prioritizing:
    1. Explicit environment variable
    2. CUDA
    3. TensorRT
    4. Core ML
    5. CPU (fallback)
    """
    available_providers = get_available_providers()
    log.info(f"Available ONNX execution providers: {available_providers}")

    # 1. Check for explicit environment variable override
    explicit_provider = os.environ.get('ONNX_EXECUTION_PROVIDER')
    if explicit_provider:
        if explicit_provider in available_providers:
            log.info(f"Using explicitly specified ONNX execution provider: {explicit_provider}")
            return [explicit_provider]
        else:
            log.warning(f"Explicitly specified provider '{explicit_provider}' not available.  Trying automatic detection...")

    # 2. Prioritize CUDA, then TensorRT, then Core ML, then CPU.
    preferred_providers = [
        'CUDAExecutionProvider',    # NVIDIA GPUs (CUDA)
        'TensorrtExecutionProvider', # NVIDIA GPUs (TensorRT)
        'CoreMLExecutionProvider',  # macOS (Apple Silicon)
        'CPUExecutionProvider',     # Fallback to CPU
    ]

    for provider in preferred_providers:
        if provider in available_providers:
            log.info(f"Using ONNX execution provider: {provider}")
            return [provider]

    log.error("No suitable ONNX execution provider found.  Falling back to CPU.  This will be very slow!")
    return ['CPUExecutionProvider'] # Explicitly return CPU as a fallback

try:
    providers = get_best_execution_provider()

    session = InferenceSession(
        "kokoro-v1.0.onnx", providers=providers, sess_options=sess_options
    )
    kokoro = Kokoro.from_session(session, "voices-v1.0.bin")
    log.info("Kokoro TTS initialized successfully")
except Exception as e:
    log.error(f"Failed to initialize Kokoro TTS: {e}")
    kokoro = None

def get_voices():
    """Get list of available voices"""
    if kokoro:
        return kokoro.get_voices()
    return ["af_heart"]  # Default fallback

def preprocess_text(text):
    """
    Preprocess text to improve TTS quality and handle problematic characters
    """
    # Replace common issues that cause phonemizer warnings
    text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)  # Add space between numbers and letters
    text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)  # Add space between letters and numbers

    # Remove or replace characters that might cause issues
    text = re.sub(r'[^\w\s.,!?\'"-:;()[\]{}]', ' ', text)  # Replace unusual chars with space

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_audio(text, voice):
    """Get audio using Kokoro TTS

    Args:
        text (str): Text to convert to speech
        voice (str): Voice ID to use

    Returns:
        tuple: (samples, sample_rate) or (None, None) if generation failed
    """
    if not kokoro:
        log.error("Kokoro TTS not initialized")
        return None, None

    try:
        # Preprocess text to avoid phonemizer warnings
        processed_text = preprocess_text(text)

        if not processed_text:
            log.warning("Empty text after preprocessing")
            return None, None

        # Generate audio
        samples, sample_rate = kokoro.create(processed_text, voice, is_phonemes=False)

        if len(samples) == 0:
            log.error("Error in generating audio - zero length output")
            # Try again with fallback voice
            if voice != "af_heart":
                log.info(f"Retrying with fallback voice af_heart")
                samples, sample_rate = kokoro.create(processed_text, "af_heart", is_phonemes=False)

            if len(samples) == 0:
                return None, None

        return samples, sample_rate

    except Exception as e:
        log.error(f"Kokoro TTS error: {e}")

        # Try again with a simpler version of the text
        try:
            # Further simplify text by removing punctuation
            simplified_text = re.sub(r'[^\w\s]', ' ', text).strip()
            simplified_text = re.sub(r'\s+', ' ', simplified_text)

            if simplified_text:
                log.info("Retrying with simplified text")
                samples, sample_rate = kokoro.create(simplified_text, voice, is_phonemes=False)
                if len(samples) > 0:
                    return samples, sample_rate

        except Exception as retry_error:
            log.error(f"Retry with simplified text failed: {retry_error}")

        return None, None
