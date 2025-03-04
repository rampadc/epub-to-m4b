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


def get_chapters(epub_book, dirs):
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
            content = filter_chapter(doc)
            if content:  # Only add non-empty chapters
                chapters.append(content)

        return chapters
    except Exception as e:
        raise DependencyError(f"Error extracting chapters: {e}")


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
    """Process a single sentence and return the result"""
    sentence_num, sentence, sentence_file, api_endpoint, api_key, model = sentence_data

    if os.path.exists(sentence_file):
        # File already exists, skip processing
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


def assemble_audiobook_m4b(chapters_dir, output_file, metadata, cover_file):
    """
    Assemble final audiobook in M4B format with chapter markers using PyDub for audio
    processing and FFmpeg for the final M4B creation.

    Args:
        chapters_dir: Directory containing chapter audio files
        output_file: Path to the final output file (should end with .m4b)
        metadata: Book metadata dictionary
        cover_file: Path to cover image file (or None)

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

        # Sort chapter files numerically
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
        chapters = get_chapters(epub_book, dirs)

        if not chapters:
            raise DependencyError("Failed to extract any chapters from the eBook")

        # Process chapters incrementally
        chapter_files = []
        current_sentence_num = 0

        # Process each chapter separately
        for chapter_num, sentences in enumerate(chapters, 1):
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
        )

        if final_audiobook:
            print(f"\nAudiobook created successfully: {final_audiobook}")
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
