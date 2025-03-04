import os
import re
import shutil
import argparse
import json
import subprocess
import sys
import requests
import ebooklib
import hashlib
from ebooklib import epub
from bs4 import BeautifulSoup
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import traceback

# Essential constants
tmp_dir = "tmp"
punctuation_list = ['.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '/', '\\', '-', '–', '—', '…']
switch_punctuations = {"'": "'", "'": "'", """: '"', """: '"', "„": '"', "‚": "'", "‟": '"', "′": "'", "″": '"'}
default_audio_proc_format = "wav"
max_tokens = 250  # Default max tokens per sentence


class DependencyError(Exception):
    def __init__(self, message=None):
        super().__init__(message)
        print(message)
        traceback.print_exc()
        sys.exit(1)


def prepare_dirs(input_file, output_dir=None):
    """Prepare directory structure for processing"""
    try:
        if output_dir is None:
            output_dir = os.path.join(tmp_dir, hashlib.md5(input_file.encode()).hexdigest())

        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        chapters_dir = os.path.join(output_dir, "chapters")
        chapters_dir_sentences = os.path.join(chapters_dir, "sentences")

        os.makedirs(chapters_dir, exist_ok=True)
        os.makedirs(chapters_dir_sentences, exist_ok=True)

        ebook_path = os.path.join(output_dir, os.path.basename(input_file))
        epub_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.epub')

        shutil.copy(input_file, ebook_path)

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
    text = re.sub(r'(?<=[\p{L}])(?=\d)|(?<=\d)(?=[\p{L}])', ' ', text)

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

        # Split by punctuation marks
        sentences = re.findall(punctuation_pattern_split, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Group sentences by token count
        return get_sentences(sentences, max_tokens)
    except Exception as e:
        print(f"Error processing document: {e}")
        return []


def get_sentences(sentences, max_tokens):
    """Group sentences to respect maximum token limits"""
    result = []
    current_sentence = ""
    current_token_count = 0

    for sentence in sentences:
        token_count = len(sentence.split())

        if current_token_count + token_count > max_tokens:
            if current_sentence:
                result.append(current_sentence.strip())
            current_sentence = sentence
            current_token_count = token_count
        else:
            current_sentence += (" " if current_sentence else "") + sentence
            current_token_count += token_count

    if current_sentence:
        result.append(current_sentence.strip())

    return result


def get_audio_from_api(text, api_endpoint, api_key):
    """Get audio from OpenAI-compatible API"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': 'mlx-community/Kokoro-82M-bf16',  # This can be parameterized
        'input': text
    }

    try:
        response = requests.post(api_endpoint, headers=headers, json=data)

        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            return None

        # Save the audio content
        return response.content
    except Exception as e:
        print(f"API request error: {e}")
        return None


def process_chapters(chapters, dirs, api_endpoint, api_key):
    """Process chapters and generate audio files"""
    sentence_number = 0
    total_sentences = sum(len(chapter) for chapter in chapters)

    print(f"Processing {len(chapters)} chapters with {total_sentences} total sentences...")

    with tqdm(total=total_sentences, desc="Converting", unit="sentence") as progress_bar:
        for chapter_num, sentences in enumerate(chapters, 1):
            chapter_sentences = []

            for sentence in sentences:
                if not sentence.strip():
                    continue

                # Filename for this sentence
                sentence_file = os.path.join(dirs["chapters_dir_sentences"],
                                             f"{sentence_number}.{default_audio_proc_format}")

                # Only process if file doesn't already exist
                if not os.path.exists(sentence_file):
                    print(f"\nProcessing: {sentence}")

                    # Get audio from API
                    audio_data = get_audio_from_api(sentence, api_endpoint, api_key)

                    if audio_data:
                        # Save the audio file
                        with open(sentence_file, 'wb') as f:
                            f.write(audio_data)
                        print(f"Saved audio to {sentence_file}")
                    else:
                        print(f"Failed to get audio for sentence {sentence_number}")

                chapter_sentences.append(sentence_file)
                sentence_number += 1
                progress_bar.update(1)

            # Combine sentence audio files for this chapter
            chapter_audio_file = os.path.join(dirs["chapters_dir"],
                                              f"chapter_{chapter_num}.{default_audio_proc_format}")
            combine_audio_sentences(chapter_sentences, chapter_audio_file)

    # Combine all chapter files into final audiobook
    chapter_files = [os.path.join(dirs["chapters_dir"], f) for f in os.listdir(dirs["chapters_dir"])
                     if f.startswith("chapter_") and f.endswith(f".{default_audio_proc_format}")]

    if chapter_files:
        output_file = os.path.join(dirs["process_dir"], f"audiobook.{default_audio_proc_format}")
        combine_audio_chapters(chapter_files, output_file)
        return output_file

    return None


def combine_audio_sentences(sentence_files, output_file):
    """Combine sentence audio files into a chapter file"""
    try:
        if not sentence_files:
            return False

        combined = AudioSegment.empty()

        for file in sentence_files:
            if os.path.exists(file):
                audio = AudioSegment.from_file(file, format=default_audio_proc_format)
                combined += audio

        combined.export(output_file, format=default_audio_proc_format)
        print(f"Combined sentences into: {output_file}")
        return True
    except Exception as e:
        print(f"Error combining sentence audio: {e}")
        return False


def combine_audio_chapters(chapter_files, output_file):
    """Combine chapter audio files into full audiobook"""
    try:
        if not chapter_files:
            return False

        combined = AudioSegment.empty()

        for file in sorted(chapter_files, key=lambda x: int(re.search(r'chapter_(\d+)', x).group(1))):
            if os.path.exists(file):
                audio = AudioSegment.from_file(file, format=default_audio_proc_format)
                combined += audio

        combined.export(output_file, format=default_audio_proc_format)
        print(f"Combined chapters into final audiobook: {output_file}")
        return True
    except Exception as e:
        print(f"Error combining chapter audio: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert eBooks to audiobooks using OpenAI-compatible API")
    parser.add_argument("--input", "-i", required=True, help="Input eBook file")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--api-endpoint", required=True, help="OpenAI-compatible API endpoint")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--max-tokens", type=int, default=250, help="Maximum tokens per sentence")

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
            shutil.copy(dirs["ebook"], dirs["epub_path"])

        # Read the EPUB
        print("Reading EPUB content...")
        epub_book = epub.read_epub(dirs["epub_path"], {'ignore_ncx': True})

        # Extract chapters and split into sentences
        chapters = get_chapters(epub_book, dirs)

        if not chapters:
            raise DependencyError("Failed to extract any chapters from the eBook")

        # Process chapters and generate audio
        output_file = process_chapters(chapters, dirs, args.api_endpoint, args.api_key)

        if output_file and os.path.exists(output_file):
            print(f"\nAudiobook created successfully: {output_file}")
            return 0
        else:
            print("\nFailed to create audiobook")
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