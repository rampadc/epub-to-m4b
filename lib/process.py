import os

from pydub.audio_segment import AudioSegment
from tqdm import tqdm
from config import DEFAULT_MAX_WORKERS, default_audio_proc_format
from lib.tts import get_audio
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_sentence(sentence_data):
    """Process a single sentence chunk and return the result"""
    sentence_num, sentence, sentence_file, api_endpoint, api_key, model = sentence_data

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
    audio_data = get_audio(sentence, api_endpoint, api_key, model)

    if audio_data:
        # Save the audio file
        with open(sentence_file, 'wb') as f:
            f.write(audio_data)
        return sentence_num, sentence_file, True
    else:
        return sentence_num, sentence_file, False

def process_chapters(chapters, dirs, api_endpoint, api_key, model, max_workers=DEFAULT_MAX_WORKERS):
    """Process chapters and generate audio files for each sentence using multiple threads"""
    sentence_number = 0
    total_sentences = sum(len(chapter) for chapter in chapters)
    audio_files = []

    # Prepare the tasks list
    tasks = []

    print(f"Processing {len(chapters)} chapters with {total_sentences} total sentences...")
    print(f"Using {max_workers} concurrent workers")

    # Create the list of sentence processing tasks
    for chapter_num, sentences in enumerate(chapters, 1):
        for sentence in sentences:
            if not sentence.strip():
                continue

            # Filename for this sentence
            sentence_file = os.path.join(dirs["chapters_dir_sentences"],
                                        f"{sentence_number}.{default_audio_proc_format}")
            audio_files.append(sentence_file)

            # Add to tasks list
            tasks.append((sentence_number, sentence, sentence_file, api_endpoint, api_key, model))
            sentence_number += 1

    # Process sentences using thread pool
    results = []
    with tqdm(total=len(tasks), desc="Converting", unit="sentence") as progress_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sentence = {
                executor.submit(process_sentence, task): task for task in tasks
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_sentence):
                task = future_to_sentence[future]
                try:
                    sentence_num, sentence_file, success = future.result()
                    if success:
                        # Successfully processed
                        results.append((sentence_num, sentence_file))
                    else:
                        print(f"\nFailed to process sentence {task[0]}: {task[1][:30]}...")
                except Exception as exc:
                    print(f"\nError processing sentence {task[0]}: {exc}")

                progress_bar.update(1)

    print(f"Generated {len(results)} individual audio files for sentences")
    return audio_files
