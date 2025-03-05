import os
import re
import subprocess
import traceback
import sys

from bs4 import BeautifulSoup
from config import tmp_dir
import hashlib
import ebooklib
import shutil
from config import switch_punctuations
from config import punctuation_list

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
                preprocess_model=None, max_chunk_size=1000, max_sentences=2, max_seconds=10.0):
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
        return get_sentences(phoneme_list)
    except Exception as e:
        print(f"Error processing document: {e}")
        return []


def get_sentences(phoneme_list):
    """
    Split a list of phonemes into proper sentences first, then respect character limits.

    Args:
        phoneme_list: A list of text fragments typically ending with punctuation

    Returns:
        A list of sentences limited to 500 characters each
    """
    # Maximum characters per chunk
    max_chars = 500

    # First, join all phonemes to get the complete text
    complete_text = ' '.join(phoneme_list)

    # Split into actual sentences using regex for period, question mark, exclamation mark
    # followed by a space and uppercase letter or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z]|$)'
    raw_sentences = re.split(sentence_pattern, complete_text)

    # Further refine sentences to respect character limit
    final_sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If sentence is already short enough, add it directly
        if len(sentence) <= max_chars:
            final_sentences.append(sentence)
        else:
            # Process sentences that exceed character limit
            # Try to break at logical points like commas, semicolons, etc.
            chunk_start = 0

            while chunk_start < len(sentence):
                # Try to find a good breaking point near the character limit
                chunk_end = min(chunk_start + max_chars, len(sentence))

                # If we're not at the end of the sentence, try to find a natural breaking point
                if chunk_end < len(sentence):
                    # Look for the last punctuation or space before the limit
                    breaking_points = [sentence.rfind(c, chunk_start, chunk_end) for c in ['. ', ', ', '; ', ' ']]
                    best_break = max(breaking_points)

                    # If found a good breaking point, use it
                    if best_break > chunk_start:
                        chunk_end = best_break + 1  # Include the breaking character

                # Extract the chunk and add it to final sentences
                chunk = sentence[chunk_start:chunk_end].strip()
                if chunk:
                    final_sentences.append(chunk)

                # Move to the next chunk
                chunk_start = chunk_end

    return final_sentences
