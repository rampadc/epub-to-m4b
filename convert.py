import os
import re
import shutil
import argparse
import subprocess
import sys
import requests
import ebooklib
import hashlib
from ebooklib import epub
from bs4 import BeautifulSoup
from pydub import AudioSegment
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Essential constants
tmp_dir = "tmp"
punctuation_list = ['.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '/', '\\', '-', '–', '—', '…']
switch_punctuations = {"'": "'", "'": "'", """: '"', """: '"', "„": '"', "‚": "'", "‟": '"', "′": "'", "″": '"'}
default_audio_proc_format = "wav"
max_tokens = 10  # Default max tokens per sentence

DEFAULT_API_ENDPOINT = "http://localhost:10240/v1"
DEFAULT_API_KEY = "xxxx"
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_MAX_WORKERS = 8  # Default number of concurrent threads

class DependencyError(Exception):
    def __init__(self, message=None):
        super().__init__(message)
        print(message)
        traceback.print_exc()
        sys.exit(1)


def get_book_metadata(epub_book):
    """
    Extract metadata from EPUB book including title, author, and other details.

    Args:
        epub_book: The epub book object

    Returns:
        Dictionary containing book metadata
    """
    metadata = {
        "title": None,
        "creator": None,
        "contributor": None,
        "language": None,
        "identifier": None,
        "publisher": None,
        "date": None,
        "description": None,
        "subject": None,
        "rights": None,
        "format": None,
        "type": None,
        "coverage": None,
        "relation": None,
        "source": None,
        "modified": None,
        "identifiers": {}
    }

    # Extract metadata from epub
    for key in metadata.keys():
        data = epub_book.get_metadata('DC', key)
        if data:
            for value, attributes in data:
                metadata[key] = value

    # Set default title from filename if not found
    if not metadata["title"]:
        metadata["title"] = os.path.splitext(os.path.basename(epub_book.file_name))[0].replace('_', ' ')

    # Mark creator as False if unknown
    if not metadata["creator"] or metadata["creator"] == 'Unknown':
        metadata["creator"] = False

    # Additional identifiers like ISBN, ASIN
    identifiers = {}
    if metadata["identifier"]:
        # Try to extract ISBN/ASIN identifiers from the identifier field
        isbn_match = re.search(r'isbn[:\s]*([0-9X-]+)', str(metadata["identifier"]), re.IGNORECASE)
        if isbn_match:
            identifiers["isbn"] = isbn_match.group(1)

        asin_match = re.search(r'asin[:\s]*([0-9A-Z]+)', str(metadata["identifier"]), re.IGNORECASE)
        if asin_match:
            identifiers["mobi-asin"] = asin_match.group(1)

    metadata["identifiers"] = identifiers

    return metadata


def get_cover(epub_book, output_dir):
    """
    Extract cover image from EPUB book.

    Args:
        epub_book: The epub book object
        output_dir: Directory to save the cover

    Returns:
        Path to the cover image or None if not found
    """
    try:
        cover_image = None
        cover_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(epub_book.file_name))[0]}.jpg")

        # First try to get the cover item
        for item in epub_book.get_items_of_type(ebooklib.ITEM_COVER):
            cover_image = item.get_content()
            break

        # If not found, try to find any image with 'cover' in the name
        if not cover_image:
            for item in epub_book.get_items_of_type(ebooklib.ITEM_IMAGE):
                if 'cover' in item.file_name.lower() or 'cover' in item.get_id().lower():
                    cover_image = item.get_content()
                    break

        if cover_image:
            with open(cover_path, 'wb') as cover_file:
                cover_file.write(cover_image)
                return cover_path

        return None
    except Exception as e:
        print(f"Error extracting cover: {e}")
        return None

def prepare_dirs(input_file, output_dir=None):
    """Prepare directory structure for processing"""
    try:
        input_file = os.path.abspath(input_file)

        if output_dir is None:
            output_dir = os.path.join(tmp_dir, hashlib.md5(input_file.encode()).hexdigest())
        else:
            output_dir = os.path.abspath(output_dir)

        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        chapters_dir = os.path.join(output_dir, "chapters")
        chapters_dir_sentences = os.path.join(chapters_dir, "sentences")

        os.makedirs(chapters_dir, exist_ok=True)
        os.makedirs(chapters_dir_sentences, exist_ok=True)

        # If the input file is already in the output directory, don't create another copy
        if os.path.dirname(input_file) == output_dir:
            ebook_path = input_file
        else:
            ebook_path = os.path.join(output_dir, os.path.basename(input_file))
            # Only copy if not already in the right place
            if not os.path.exists(ebook_path) or not os.path.samefile(input_file, ebook_path):
                shutil.copy(input_file, ebook_path)

        # Create a separate path for the epub version if needed
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        epub_path = os.path.join(output_dir, f"{base_name}.epub")

        return {
            "process_dir": output_dir,
            "chapters_dir": chapters_dir,
            "chapters_dir_sentences": chapters_dir_sentences,
            "ebook": ebook_path,
            "epub_path": epub_path
        }
    except Exception as e:
        raise DependencyError(f"Error preparing directories: {e}")


def convert_to_epub(input_file, output_file):
    """Convert an ebook to EPUB format using Calibre"""
    try:
        util_app = shutil.which('ebook-convert')
        if not util_app:
            raise DependencyError("The 'ebook-convert' utility is not installed or not found.")

        print(f"Running: {util_app} {input_file} {output_file} --input-encoding=utf-8 --output-profile=generic_eink")

        result = subprocess.run(
            [util_app, input_file, output_file, '--input-encoding=utf-8', '--output-profile=generic_eink'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        if result.returncode != 0:
            raise DependencyError(f"Error converting to EPUB: {result.stderr}")

        return True
    except subprocess.CalledProcessError as e:
        raise DependencyError(f"Subprocess error: {e.stderr}")
    except Exception as e:
        raise DependencyError(f"Error converting to EPUB: {e}")


def normalize_text(text):
    """Normalize text for TTS processing"""
    # Replace problem punctuations
    for original, replacement in switch_punctuations.items():
        text = text.replace(original, replacement)

    # Replace NBSP with a normal space
    text = text.replace("\xa0", " ")

    # Replace multiple newlines with periods
    text = re.sub('(\r\n|\n\n|\r\r|\n\r)+', lambda m: ' . ' * (m.group().count("\n") // 2 + m.group().count("\r") // 2),
                  text)

    # Replace single newlines with spaces
    text = re.sub(r'[\r\n]', ' ', text)

    # Replace tabs with spaces
    text = re.sub(r'\t+', lambda m: ' ' * len(m.group()), text)

    # Add space between letters and numbers
    try:
        # This uses the regex module which supports Unicode properties
        text = re.sub(r'(?<=[\p{L}])(?=\d)|(?<=\d)(?=[\p{L}])', ' ', text)
    except:
        # Fallback if regex module fails
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_chapters(epub_book, dirs, preprocess_llm=False, api_endpoint=None, api_key=None,
                preprocess_model=None, max_chunk_size=4000, max_sentences=2, max_seconds=30.0):
    """Extract chapters from EPUB and split into sentences"""
    try:
        all_docs = list(epub_book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        if not all_docs:
            raise DependencyError("No document items found in EPUB")

        # Skip the first document (usually metadata or cover)
        all_docs = all_docs[1:]

        print("Extracting text from EPUB documents...")
        chapters = []

        for doc in all_docs:
            if preprocess_llm:
                print(f"Processing document with LLM: {doc.get_name()}")
                content = filter_chapter_with_llm(
                    doc, api_endpoint, api_key, preprocess_model,
                    max_chunk_size, max_sentences, max_seconds
                )
            else:
                content = filter_chapter(doc)

            if content:  # Only add non-empty chapters
                chapters.append(content)

        return chapters
    except Exception as e:
        raise DependencyError(f"Error extracting chapters: {e}")

def filter_chapter_with_llm(doc, api_endpoint, api_key, model, max_chunk_size=4000,
                           max_sentences=2, max_seconds=30.0):
    """Extract, normalize, and preprocess text from a document using LLM"""
    try:
        soup = BeautifulSoup(doc.get_body_content(), 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        text = soup.get_text().strip()

        # Basic normalization
        text = normalize_text(text)

        # Use LLM to preprocess and split into sentences
        sentences = preprocess_chapter_with_llm(text, api_endpoint, api_key, model, max_chunk_size)

        # Further split into appropriately sized chunks
        final_chunks = []
        current_chunk = ""
        sentence_count = 0

        for sentence in sentences:
            # Special pause indicator
            if sentence == "[PAUSE]":
                if current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = ""
                    sentence_count = 0
                final_chunks.append("[PAUSE]")
                continue

            # Estimate audio duration
            estimated_duration = len(sentence) / 5 / 3  # chars/avg word length/words per second

            if (sentence_count >= max_sentences) or \
               (estimated_duration > max_seconds * 0.7) or \
               (current_chunk and (len(current_chunk) + len(sentence))/5/3 > max_seconds):

                if current_chunk:
                    final_chunks.append(current_chunk)
                current_chunk = sentence
                sentence_count = 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                sentence_count += 1

        # Add any remaining text
        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks
    except Exception as e:
        print(f"Error processing document with LLM: {e}")
        traceback.print_exc()
        return []

def filter_chapter(doc):
    """Extract and normalize text from a document"""
    try:
        soup = BeautifulSoup(doc.get_body_content(), 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        text = soup.get_text().strip()

        # Normalize the text
        text = normalize_text(text)

        # Create regex pattern to split by sentences
        escaped_punctuation = re.escape(''.join(punctuation_list))
        punctuation_pattern_split = rf'(\S.*?[{"".join(escaped_punctuation)}])|\S+'

        # Split by punctuation marks while keeping the punctuation at the end of each word
        phoneme_list = re.findall(punctuation_pattern_split, text)
        phoneme_list = [phoneme.strip() for phoneme in phoneme_list if phoneme.strip()]

        # Group sentences by token count
        return get_sentences(phoneme_list, max_tokens)
    except Exception as e:
        print(f"Error processing document: {e}")
        return []


def get_sentences(phoneme_list, max_tokens):
    """
    Split a list of phonemes into proper sentences first, then respect token limits.

    Args:
        phoneme_list: A list of text fragments typically ending with punctuation
        max_tokens: Maximum number of tokens (words) allowed per sentence

    Returns:
        A list of sentences
    """
    # First, join all phonemes to get the complete text
    complete_text = ' '.join(phoneme_list)

    # Split into actual sentences using regex for period, question mark, exclamation mark
    # followed by a space and uppercase letter or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z]|$)'
    raw_sentences = re.split(sentence_pattern, complete_text)

    # Further refine sentences to respect max token limit
    final_sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Split long sentences based on token count
        words = sentence.split()
        if len(words) <= max_tokens:
            final_sentences.append(sentence)
        else:
            # Process sentences that exceed max token count
            current_chunk = []
            current_token_count = 0

            for word in words:
                if current_token_count + 1 > max_tokens:
                    # Add current chunk as a sentence
                    final_sentences.append(' '.join(current_chunk))
                    # Start a new chunk
                    current_chunk = [word]
                    current_token_count = 1
                else:
                    current_chunk.append(word)
                    current_token_count += 1

            # Add the last chunk if it exists
            if current_chunk:
                final_sentences.append(' '.join(current_chunk))

    return final_sentences

def get_audio_from_api(text, api_endpoint, api_key, model):
    """Get audio from OpenAI-compatible API"""
    # Construct the proper endpoint URL for audio generation
    if api_endpoint.endswith('/v1'):
        api_endpoint = f"{api_endpoint}/audio/speech"
    elif not api_endpoint.endswith('/audio/speech'):
        api_endpoint = f"{api_endpoint.rstrip('/')}/v1/audio/speech"

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': model,
        'input': text
    }

    try:
        response = requests.post(api_endpoint, headers=headers, json=data)

        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            return None

        # Return the audio content
        return response.content
    except Exception as e:
        print(f"API request error: {e}")
        return None

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
    audio_data = get_audio_from_api(sentence, api_endpoint, api_key, model)

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


def combine_audio_sentences(sentence_files, output_file):
    """Combine sentence audio files into a chapter file using PyDub"""
    try:
        if not sentence_files:
            return False

        combined = AudioSegment.empty()

        for file in sentence_files:
            if os.path.exists(file):
                audio = AudioSegment.from_file(file, format=default_audio_proc_format)
                combined += audio

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Export the combined audio
        combined.export(output_file, format=default_audio_proc_format)
        print(f"Combined sentences into: {output_file}")
        return True
    except Exception as e:
        print(f"Error combining sentence audio: {e}")
        traceback.print_exc()
        return False

# Collect timing information
def get_audio_duration(audio_file):
    """Get the duration of an audio file in milliseconds"""
    try:
        audio = AudioSegment.from_file(audio_file)
        return len(audio)
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0

# Build the text-audio mapping
def build_text_audio_mapping(chapters, sentence_data, dirs):
    """
    Build a mapping of text fragments to audio timestamps.

    Args:
        chapters: List of chapter sentences
        sentence_data: List of dictionaries with sentence information
        dirs: Directory structure dictionary

    Returns:
        Path to the generated mapping file
    """
    try:
        mapping = {
            "version": "1.0",
            "book_title": "Unknown",  # This could be populated from metadata
            "fragments": [],
            "chapters": []
        }

        current_time_ms = 0
        chapter_start_times = {}

        # First pass: calculate all durations and timestamps
        print("Calculating audio timings for synchronization...")
        for sentence_info in tqdm(sentence_data, desc="Processing timings"):
            chapter_num = sentence_info['chapter']
            sentence_file = sentence_info['file']

            if not os.path.exists(sentence_file):
                continue

            # Record chapter start time if this is the first sentence in the chapter
            if chapter_num not in chapter_start_times:
                chapter_start_times[chapter_num] = current_time_ms

            # Get audio duration
            try:
                audio = AudioSegment.from_file(sentence_file, format=default_audio_proc_format)
                duration_ms = len(audio)

                # Add to fragments list
                fragment = {
                    "chapter": chapter_num,
                    "text": sentence_info['text'],
                    "start_time": current_time_ms,
                    "end_time": current_time_ms + duration_ms,
                }

                mapping["fragments"].append(fragment)

                # Update current time
                current_time_ms += duration_ms

            except Exception as e:
                print(f"Error processing timing for sentence {sentence_info['sentence_num']}: {e}")

        # Second pass: build chapter information
        for chapter_num in sorted(chapter_start_times.keys()):
            start_time = chapter_start_times[chapter_num]

            # Find the end time of this chapter (start of next chapter or end of book)
            if chapter_num + 1 in chapter_start_times:
                end_time = chapter_start_times[chapter_num + 1]
            else:
                end_time = current_time_ms

            # Count fragments in this chapter
            fragments_in_chapter = sum(1 for f in mapping["fragments"] if f["chapter"] == chapter_num)

            mapping["chapters"].append({
                "number": chapter_num,
                "title": f"Chapter {chapter_num}",
                "start_time": start_time,
                "end_time": end_time,
                "fragment_count": fragments_in_chapter
            })

        # Calculate total duration
        mapping["total_duration_ms"] = current_time_ms

        # Save the mapping to a JSON file
        mapping_file = os.path.join(dirs["process_dir"], "text_audio_mapping.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print(f"Created text-audio mapping file with {len(mapping['fragments'])} fragments across {len(mapping['chapters'])} chapters")
        return mapping_file

    except Exception as e:
        print(f"Error building text-audio mapping: {e}")
        traceback.print_exc()
        return None

def assemble_audiobook_m4b(chapters_dir, output_file, metadata, cover_file, mapping_file=None):
    """
    Assemble final audiobook in M4B format with chapter markers using PyDub for audio
    processing and FFmpeg for the final M4B creation.

    Args:
        chapters_dir: Directory containing chapter audio files
        output_file: Path to the final output file (should end with .m4b)
        metadata: Book metadata dictionary
        cover_file: Path to cover image file (or None)
        mapping_file: Path to text-audio synchronization mapping file (or None)

    Returns:
        Path to the final audiobook file if successful, None otherwise
    """
    try:
        # Check if ffmpeg is installed
        if not shutil.which('ffmpeg'):
            print("FFmpeg is required for M4B creation with chapter markers.")
            return None

        # Get all chapter files
        chapter_files = [
            os.path.join(chapters_dir, f)
            for f in os.listdir(chapters_dir)
            if f.endswith(f'.{default_audio_proc_format}') and 'chapter_' in f
        ]

        if not chapter_files:
            print("No chapter files found!")
            return None

        # Sort chapter files numerically using our safer function
        def get_chapter_num(filename):
            match = re.search(r'chapter_(\d+)', os.path.basename(filename))
            if match:
                return int(match.group(1))
            # Default to a large number for any files without proper naming
            return float('inf')

        chapter_files.sort(key=get_chapter_num)

        print(f"Processing {len(chapter_files)} chapters...")

        # First combine all audio using PyDub
        process_dir = os.path.dirname(chapters_dir)
        combined_audio_file = os.path.join(process_dir, "combined_audio.wav")

        # Combine all chapter files into one audio file
        combined = AudioSegment.empty()

        # Store chapter timestamps for creating chapter markers
        chapter_timestamps = []
        current_pos_ms = 0

        for idx, chapter_file in enumerate(chapter_files):
            chapter_audio = AudioSegment.from_file(chapter_file, format=default_audio_proc_format)
            chapter_duration_ms = len(chapter_audio)

            # Store the timestamp information for this chapter
            chapter_timestamps.append({
                'number': idx + 1,
                'start_ms': current_pos_ms,
                'end_ms': current_pos_ms + chapter_duration_ms
            })

            # Update current position
            current_pos_ms += chapter_duration_ms

            # Add to combined audio
            combined += chapter_audio

            print(f"  Added Chapter {idx + 1} ({chapter_duration_ms/1000:.2f} seconds)")

        # Export the combined audio
        combined.export(combined_audio_file, format=default_audio_proc_format)
        print(f"Combined all chapters into temporary file: {combined_audio_file}")

        # Create FFmpeg metadata file with chapter markers
        metadata_file = os.path.join(process_dir, "metadata.txt")

        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(";FFMETADATA1\n")

            # Book metadata
            if metadata.get('title'):
                f.write(f"title={metadata['title']}\n")
            if metadata.get('creator'):
                f.write(f"artist={metadata['creator']}\n")
            if metadata.get('language'):
                f.write(f"language={metadata['language']}\n")
            if metadata.get('publisher'):
                f.write(f"publisher={metadata['publisher']}\n")
            if metadata.get('description'):
                f.write(f"description={metadata['description']}\n")

            # Extract year from publication date if available
            if metadata.get('date'):
                year_match = re.search(r'\b(19|20)\d{2}\b', str(metadata['date']))
                if year_match:
                    f.write(f"year={year_match.group(0)}\n")
                else:
                    from datetime import datetime
                    f.write(f"year={datetime.now().year}\n")

            # Add identifiers
            if metadata.get('identifiers'):
                if metadata['identifiers'].get('isbn'):
                    f.write(f"isbn={metadata['identifiers']['isbn']}\n")
                if metadata['identifiers'].get('mobi-asin'):
                    f.write(f"asin={metadata['identifiers']['mobi-asin']}\n")

            # If we have a mapping file, add a custom metadata field to reference it
            if mapping_file and os.path.exists(mapping_file):
                mapping_filename = os.path.basename(mapping_file)
                f.write(f"comment=This audiobook includes text synchronization data: {mapping_filename}\n")

            # Add chapter markers
            for chapter in chapter_timestamps:
                f.write("\n[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={chapter['start_ms']}\n")
                f.write(f"END={chapter['end_ms']}\n")
                f.write(f"title=Chapter {chapter['number']}\n")

        # Ensure output file has .m4b extension
        if not output_file.lower().endswith('.m4b'):
            output_file = f"{os.path.splitext(output_file)[0]}.m4b"

        # Prepare FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', combined_audio_file,
            '-i', metadata_file
        ]

        # Add cover image if available
        if cover_file and os.path.exists(cover_file):
            ffmpeg_cmd.extend(['-i', cover_file])

            # Map audio and cover
            ffmpeg_cmd.extend([
                '-map', '0:a',
                '-map', '2:v',
                '-disposition:v', 'attached_pic'
            ])
        else:
            # Map just audio
            ffmpeg_cmd.extend(['-map', '0:a'])

        # Add output parameters for M4B
        ffmpeg_cmd.extend([
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '44100',
            '-movflags', '+faststart',
            '-map_metadata', '1',
            '-y',
            output_file
        ])

        # Run FFmpeg command
        print("Creating M4B file with chapter markers using FFmpeg...")
        print(f"Command: {' '.join(ffmpeg_cmd)}")

        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        if process.stdout:
            for line in process.stdout:
                # Print only important FFmpeg messages
                if 'size=' in line or 'time=' in line or 'speed=' in line:
                    print(f"\rFFmpeg: {line.strip()}", end='')

        process.wait()
        print()  # New line after FFmpeg progress

        if process.returncode == 0:
            print(f"Successfully created M4B audiobook: {output_file}")

            # If we have a mapping file, copy it to be alongside the final output
            if mapping_file and os.path.exists(mapping_file):
                mapping_dest = f"{os.path.splitext(output_file)[0]}_sync.json"
                shutil.copy(mapping_file, mapping_dest)
                print(f"Copied synchronization data to: {mapping_dest}")

            # Clean up temporary files
            os.remove(combined_audio_file)
            os.remove(metadata_file)

            return output_file
        else:
            print(f"Error creating M4B file. FFmpeg exited with code {process.returncode}")
            return None

    except Exception as e:
        print(f"Error assembling M4B audiobook: {e}")
        traceback.print_exc()
        return None

def combine_audio_chapters(chapter_files, output_file):
    """Combine chapter audio files into full audiobook"""
    try:
        if not chapter_files:
            return False

        combined = AudioSegment.empty()

        # Sort chapter files by chapter number in a more robust way
        def get_chapter_num(filename):
            match = re.search(r'chapter_(\d+)', filename)
            if match:
                return int(match.group(1))
            # Default to a large number for any files without proper naming
            return float('inf')

        for file in sorted(chapter_files, key=get_chapter_num):
            if os.path.exists(file):
                audio = AudioSegment.from_file(file, format=default_audio_proc_format)
                combined += audio

        combined.export(output_file, format="mp3", bitrate="128k")
        print(f"Combined chapters into final audiobook: {output_file}")
        return True
    except Exception as e:
        print(f"Error combining chapter audio: {e}")
        return False

def preprocess_chapter_with_llm(chapter_text, api_endpoint, api_key, model, max_chunk_size=4000):
    """
    Use LLM to preprocess chapter text for better TTS rendering.
    Handles long chapters by splitting into manageable chunks.

    Args:
        chapter_text: The chapter text to preprocess
        api_endpoint: API endpoint for the LLM
        api_key: API key for authentication
        model: LLM model to use
        max_chunk_size: Maximum chunk size in characters

    Returns:
        Preprocessed chapter text optimized for TTS and properly split into sentences
    """
    # Construct the proper endpoint URL for chat completion
    if api_endpoint.endswith('/v1'):
        chat_endpoint = f"{api_endpoint}/chat/completions"
    else:
        chat_endpoint = f"{api_endpoint.rstrip('/')}/v1/chat/completions"

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # Create the prompt for preprocessing with improved TTS safety
    system_prompt = """You are a text preparation expert for text-to-speech (TTS) systems.
    Your job is to format text for natural, high-quality audio narration while ensuring it's 100% TTS-safe.

    Guidelines:
    1. Properly split sentences at logical points where a narrator would pause
    2. Convert Roman numerals to their spoken equivalent (e.g., "Chapter IV" → "Chapter 4")
    3. Format numbers, dates, measurements, and currencies for natural reading
    4. Handle abbreviations, acronyms, and special characters appropriately
    5. Add strategic commas for natural pausing
    6. Format dialog with appropriate attributions and pauses
    7. Handle special textual elements like chapter headings, quotes, and lists

    TTS SAFETY RULES (CRITICAL):
    1. Normalize all quotation marks: Replace curly quotes (", ", ', ') with straight quotes (' and ")
    2. Remove or replace problematic characters: \, `, |, *, ~, <, >, ^
    3. Replace em dashes (—) with regular dashes (-) and add spaces around them
    4. For dialog, verbalize quote indicators when needed: e.g., "open quote", "close quote"
    5. Remove any HTML or XML-like tags that might remain in the text
    6. Express mathematical formulas and equations in spoken form
    7. Avoid consecutive punctuation marks (use only one)
    8. Expand symbols like % and @ to "percent" and "at"
    9. Ensure proper spacing after periods, commas, and other punctuation
    10. Add a pause indicator (comma) before dialog attribution (e.g., "Hello," he said)
    11. Ensure no unmatched brackets () [] {} remain in text

    For each chunk of text I provide, return:
    - Properly structured sentences, each on its own line
    - Maintain paragraph breaks with blank lines
    - Only return the processed text, no explanations
    """

    user_prompt = "Process this text for optimal text-to-speech narration. Split into proper sentences and format for natural speech, ensuring the text is completely safe for TTS processing:"

    # Split long chapter into manageable chunks
    chunks = []
    paragraphs = re.split(r'\n\s*\n', chapter_text)
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed our limit, finalize current chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)

    processed_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")

        data = {
            'model': model,
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\n{chunk}"}
            ],
            'temperature': 0.3
        }

        try:
            response = requests.post(chat_endpoint, headers=headers, json=data)

            if response.status_code != 200:
                print(f"LLM preprocessing error: {response.status_code} - {response.text}")
                # Perform basic safety sanitization ourselves if LLM fails
                safe_chunk = sanitize_for_tts(chunk)
                processed_chunks.append(safe_chunk)
                continue

            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                processed_text = result['choices'][0]['message']['content'].strip()
                # Double-check safety of LLM output
                processed_text = sanitize_for_tts(processed_text)
                processed_chunks.append(processed_text)
            else:
                print("LLM returned unexpected response format")
                safe_chunk = sanitize_for_tts(chunk)
                processed_chunks.append(safe_chunk)

        except Exception as e:
            print(f"LLM preprocessing error: {e}")
            safe_chunk = sanitize_for_tts(chunk)
            processed_chunks.append(safe_chunk)

    # Join processed chunks
    processed_text = "\n\n".join(processed_chunks)

    # Extract sentences and further split them into appropriate TTS chunks
    raw_sentences = []
    for paragraph in processed_text.split("\n\n"):
        # Split on line breaks as these should now represent sentence boundaries
        for line in paragraph.split("\n"):
            line = line.strip()
            if line:
                raw_sentences.append(line)

        # Add a small pause between paragraphs
        if raw_sentences and raw_sentences[-1] != "":
            raw_sentences.append("")

    # Further split sentences into appropriately sized TTS chunks
    final_sentences = []
    current_tts_chunk = ""
    sentence_count = 0

    for sentence in raw_sentences:
        # Estimate audio duration: ~3 words per second is typical narration speed
        # Average English word is ~5 characters
        estimated_duration = len(sentence) / 5 / 3  # in seconds

        if sentence == "":  # Paragraph break
            if current_tts_chunk:
                final_sentences.append(current_tts_chunk)
                current_tts_chunk = ""
                sentence_count = 0
            final_sentences.append("")
            continue

        # Start a new chunk if:
        # 1. We already have 2 sentences OR
        # 2. Adding this sentence would make the chunk too long (>30 seconds) OR
        # 3. This single sentence is already very long (>20 seconds)
        if (sentence_count >= 2) or \
           (current_tts_chunk and estimated_duration > 20) or \
           (len(current_tts_chunk) > 0 and len(current_tts_chunk) + len(sentence) > 800):  # ~800 chars ≈ 30 seconds

            if current_tts_chunk:
                final_sentences.append(current_tts_chunk)
                current_tts_chunk = sentence
                sentence_count = 1
            else:
                # If a single sentence is very long, we still need to include it
                final_sentences.append(sentence)
                sentence_count = 0
                current_tts_chunk = ""
        else:
            # Add to current chunk
            if current_tts_chunk:
                current_tts_chunk += " " + sentence
            else:
                current_tts_chunk = sentence
            sentence_count += 1

    # Add any remaining text
    if current_tts_chunk:
        final_sentences.append(current_tts_chunk)

    # Remove any trailing empty sentences
    while final_sentences and not final_sentences[-1]:
        final_sentences.pop()

    # Ensure we don't have empty entries in the middle (convert to short pause indicator)
    final_sentences = [s if s else "[PAUSE]" for s in final_sentences]

    return final_sentences

def sanitize_for_tts(text):
    """
    Perform basic sanitization of text to make it safer for TTS processing.
    This is a fallback in case LLM processing fails.
    """
    # Normalize quotation marks
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Replace problematic characters
    text = text.replace('\\', ' backslash ').replace('`', "'")
    text = text.replace('|', ' or ').replace('*', ' star ')
    text = text.replace('~', ' approximately ').replace('^', ' caret ')

    # Fix dashes
    text = text.replace('—', ' - ').replace('–', ' - ')

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix spacing after punctuation
    text = re.sub(r'([.,;:!?])(\w)', r'\1 \2', text)

    # Remove HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)

    # Ensure proper bracket pairing by removing unmatched brackets
    # This is a simplistic approach - LLM does a better job at this
    for bracket_pair in [('(', ')'), ('[', ']'), ('{', '}')]:
        if text.count(bracket_pair[0]) != text.count(bracket_pair[1]):
            text = text.replace(bracket_pair[0], ' ').replace(bracket_pair[1], ' ')

    # Convert percentage symbols
    text = re.sub(r'(\d+)%', r'\1 percent', text)

    # Convert @ symbol
    text = text.replace('@', ' at ')

    # Clean up consecutive punctuation
    text = re.sub(r'([.,;:!?]){2,}', r'\1', text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Convert eBooks to audiobooks using OpenAI-compatible API")
    parser.add_argument("--input", "-i", required=True, help="Input eBook file")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--api-endpoint", default=DEFAULT_API_ENDPOINT,
                        help=f"OpenAI-compatible API endpoint (default: {DEFAULT_API_ENDPOINT})")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                        help=f"API key for authentication (default: {DEFAULT_API_KEY})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use for TTS (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-tokens", type=int, default=250, help="Maximum tokens per sentence (default: 250)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Maximum number of concurrent API requests (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav", "ogg", "flac"],
                        help="Output format (default: mp3)")
    parser.add_argument("--sync-text", action="store_true",
                        help="Generate text-audio synchronization data for text highlighting")
    parser.add_argument("--preprocess-llm", action="store_true",
                        help="Use LLM to preprocess chapter text before TTS conversion")
    parser.add_argument("--preprocess-model", default="mlx-community/Llama-3.2-1B-Instruct-4bit",
                        help="Model to use for text preprocessing (default: mlx-community/Llama-3.2-1B-Instruct-4bit)")
    parser.add_argument("--preprocess-max-chunk", type=int, default=4000,
                        help="Maximum chunk size in characters for LLM preprocessing (default: 4000)")
    parser.add_argument("--max-seconds-per-chunk", type=float, default=30.0,
                       help="Maximum estimated length in seconds for TTS audio chunks (default: 30)")
    parser.add_argument("--max-sentences-per-chunk", type=int, default=2,
                       help="Maximum number of sentences per TTS chunk (default: 2)")
    parser.add_argument("--chapter-start", type=int, default=1,
                        help="Starting chapter number to process (default: 1)")
    parser.add_argument("--chapter-end", type=int, default=None,
                        help="Ending chapter number to process (inclusive, default: process all chapters)")

    args = parser.parse_args()

    global max_tokens
    max_tokens = args.max_tokens

    try:
        # Prepare directories
        dirs = prepare_dirs(args.input, args.output_dir)

        # Convert to EPUB if needed
        if not args.input.lower().endswith('.epub'):
            print("Converting input file to EPUB format...")
            convert_to_epub(dirs["ebook"], dirs["epub_path"])
        else:
            # If input is already EPUB, just copy it
            input_path = os.path.abspath(args.input)
            epub_path = os.path.abspath(dirs["epub_path"])

            if input_path != epub_path:
                shutil.copy(input_path, epub_path)

            # Always use the file in the processing directory
            dirs["epub_path"] = epub_path

        # Read the EPUB
        print("Reading EPUB content...")
        epub_book = epub.read_epub(dirs["epub_path"], {'ignore_ncx': True})

        # Extract metadata and cover
        print("Extracting book metadata...")
        metadata = get_book_metadata(epub_book)
        cover_file = get_cover(epub_book, dirs["process_dir"])

        if cover_file:
            print(f"Extracted cover: {cover_file}")
        else:
            print("No cover image found in the book")

        # Extract chapters and split into sentences
        chapters = get_chapters(
            epub_book,
            dirs,
            preprocess_llm=args.preprocess_llm,
            api_endpoint=args.api_endpoint,
            api_key=args.api_key,
            preprocess_model=args.preprocess_model,
            max_chunk_size=args.preprocess_max_chunk
        )

        # Filter chapters based on start/end arguments if specified
        if args.chapter_start > 1 or args.chapter_end is not None:
            original_chapter_count = len(chapters)
            start_idx = args.chapter_start - 1  # Convert to 0-based index
            end_idx = args.chapter_end if args.chapter_end is not None else len(chapters)

            # Validate chapter range
            if start_idx < 0 or start_idx >= len(chapters):
                print(f"Error: Starting chapter {args.chapter_start} is out of range (book has {len(chapters)} chapters)")
                return 1
            if end_idx > len(chapters):
                print(f"Warning: Ending chapter {args.chapter_end} exceeds available chapters. Using last chapter ({len(chapters)}) instead.")
                end_idx = len(chapters)

            # Select only the specified chapters
            chapters = chapters[start_idx:end_idx]
            print(f"Processing chapters {args.chapter_start} to {end_idx} (out of {original_chapter_count} total chapters)")

        if not chapters:
            raise DependencyError("Failed to extract any chapters from the eBook")

        # Process chapters incrementally
        chapter_files = []
        all_sentence_files = []  # Store all sentence files for mapping
        all_sentences_data = []  # Store mapping between sentences and their text content
        current_sentence_num = 0

        # Process each chapter separately
        for chapter_idx, sentences in enumerate(chapters):
            chapter_num = chapter_idx + args.chapter_start
            valid_sentences = [s for s in sentences if s.strip()]
            if not valid_sentences:
                continue

            print(f"\nProcessing Chapter {chapter_num} with {len(valid_sentences)} sentences...")

            # Process this chapter's sentences and get audio files
            chapter_audio_files = []

            for sentence in valid_sentences:
                # Filename for this sentence
                sentence_file = os.path.join(dirs["chapters_dir_sentences"],
                                            f"{current_sentence_num}.{default_audio_proc_format}")
                chapter_audio_files.append((current_sentence_num, sentence, sentence_file))

                # Store for mapping
                all_sentence_files.append(sentence_file)
                all_sentences_data.append({
                    'chapter': chapter_num,
                    'sentence_num': current_sentence_num,
                    'text': sentence,
                    'file': sentence_file
                })

                current_sentence_num += 1

            # Process sentences using thread pool
            with tqdm(total=len(chapter_audio_files), desc=f"Chapter {chapter_num}", unit="sentence") as progress_bar:
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    futures = []

                    for sentence_num, sentence, sentence_file in chapter_audio_files:
                        if os.path.exists(sentence_file):
                            progress_bar.update(1)
                            continue

                        future = executor.submit(
                            process_sentence,
                            (sentence_num, sentence, sentence_file, args.api_endpoint, args.api_key, args.model)
                        )
                        futures.append(future)

                    for future in as_completed(futures):
                        try:
                            _, _, success = future.result()
                            if not success:
                                print(f"\nFailed to process a sentence in Chapter {chapter_num}")
                        except Exception as exc:
                            print(f"\nError processing sentence: {exc}")

                        progress_bar.update(1)

            # Combine this chapter's sentences into a chapter file
            sentence_files = [s[2] for s in chapter_audio_files]
            chapter_file = os.path.join(dirs["chapters_dir"], f"chapter_{chapter_num}.{default_audio_proc_format}")

            print(f"Combining sentences for Chapter {chapter_num}...")
            if combine_audio_sentences(sentence_files, chapter_file):
                chapter_files.append(chapter_file)
                print(f"  Created chapter audio: {chapter_file}")
            else:
                print(f"  Failed to create chapter audio for Chapter {chapter_num}")

        # Generate text-audio synchronization data if requested
        mapping_file = None
        if args.sync_text:
            print("\nBuilding text-audio synchronization data...")
            mapping_file = build_text_audio_mapping(chapters, all_sentences_data, dirs)

        # Create output audiobook file
        output_base = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(dirs["process_dir"], f"{output_base}.{args.format}")

        # Assemble final audiobook with chapters
        print("\nAssembling final audiobook with chapters...")
        final_audiobook = assemble_audiobook_m4b(
            dirs["chapters_dir"],
            output_file,
            metadata,
            cover_file,
            mapping_file if args.sync_text else None
        )

        if final_audiobook:
            print(f"\nAudiobook created successfully: {final_audiobook}")
            if mapping_file:
                print(f"Text-audio synchronization data: {mapping_file}")
                print("\nNote: To use text highlighting with audio playback, you'll need a compatible player that supports this mapping format.")
            return 0
        else:
            print("\nFailed to create audiobook with chapters")
            return 1

    except DependencyError as e:
        # Already handled in the exception
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
