import os
import re
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from .preprocess import normalize_text, get_sentences  # Import from .preprocess
from pydub import AudioSegment
from config import default_audio_proc_format
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class ChapterManager:
    def __init__(self, epub_path, output_dir, default_voice='af_heart'):
        self.epub_path = Path(epub_path)
        self.output_dir = Path(output_dir)
        self.chapters_dir = self.output_dir / "chapters"
        self.sentences_dir = self.chapters_dir / "sentences"
        self.default_voice = default_voice
        self.book = None
        self._load_epub()
        self.chapter_data = []  # List of dictionaries: {'number': int, 'title': str, 'sentences': [str], 'audio_files': [Path]}
        self._extract_chapters()


    def _load_epub(self):
        try:
            self.book = epub.read_epub(self.epub_path)
            log.info(f"Successfully read EPUB: {self.epub_path}")
        except Exception as e:
            log.error(f"Error reading EPUB: {e}")
            raise

    def _extract_chapters(self):
        """Extracts chapters, normalizes text, and splits into sentences."""
        if not self.book:
            return

        all_docs = list(self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        all_docs = all_docs[1:]  # Skip the first document

        chapter_number = 1
        for doc in all_docs:
            try:
                soup = BeautifulSoup(doc.get_body_content(), 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text().strip()
                normalized_text = normalize_text(text)
                sentences = get_sentences(normalized_text.split())

                # Get chapter title from HTML content (if available)
                title_tag = soup.find(['h1', 'h2', 'h3'])  # Common title tags
                chapter_title = title_tag.get_text().strip() if title_tag else f"Chapter {chapter_number}"

                self.chapter_data.append({
                    'number': chapter_number,
                    'title': chapter_title,
                    'sentences': sentences,
                    'audio_files': [],  # Initially empty, populated during processing
                    'html_file': doc.file_name
                })
                chapter_number += 1
            except Exception as e:
                log.error(f"Error processing document {doc.file_name}: {e}")
                continue # Continue with the next document instead of crashing

    def get_chapter_audio_filepath(self, chapter_number):
        return self.chapters_dir / f"chapter_{chapter_number}.{default_audio_proc_format}"

    def get_sentence_audio_filepath(self, chapter_number, sentence_number):
        return self.sentences_dir / f"chapter_{chapter_number}_sentence_{sentence_number}.{default_audio_proc_format}"


    def prepare_directories(self):
        """Creates the necessary output directories."""
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        self.sentences_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Prepared directories: {self.chapters_dir}, {self.sentences_dir}")


    def get_all_sentence_data(self, voice_map=None):
        """Returns a list of all sentences with associated data for processing."""
        all_sentences = []

        for chapter_info in self.chapter_data:
            chapter_num = chapter_info['number']
            # Determine the voice for this chapter
            voice = self.default_voice  # Default voice
            if voice_map and chapter_num in voice_map:
                voice = voice_map[chapter_num]  # Override with chapter-specific voice


            for i, sentence in enumerate(chapter_info['sentences']):
                sentence_file = self.get_sentence_audio_filepath(chapter_num, i)
                all_sentences.append({
                    'chapter': chapter_num,
                    'sentence_num': i,
                    'text': sentence,
                    'file': sentence_file,
                    'voice': voice
                })
        return all_sentences


    def combine_sentences_to_chapter(self, chapter_number):
        """Combines sentence audio files for a given chapter."""
        chapter_info = self.get_chapter_info(chapter_number)
        if not chapter_info:
            log.error(f"Chapter {chapter_number} not found.")
            return False

        output_file = self.get_chapter_audio_filepath(chapter_number)
        sentence_files = [data['file'] for data in self.get_all_sentence_data() if data['chapter'] == chapter_number]

        # Ensure sentence files exist
        valid_sentence_files = [f for f in sentence_files if f.exists()]
        if not valid_sentence_files:
            log.warning(f"No sentence audio files found for chapter {chapter_number}.")
            return False
        if len(valid_sentence_files) < len(sentence_files):
            log.warning(f"Some sentence audio files missing for chapter {chapter_number}.")


        combined = AudioSegment.empty()
        for file in valid_sentence_files:
            try:
                audio = AudioSegment.from_file(file, format=default_audio_proc_format)
                combined += audio
            except Exception as e:
                log.error(f"Error loading sentence audio {file}: {e}")
                return False # Critical error, stop combining

        try:
            combined.export(output_file, format=default_audio_proc_format)
            log.info(f"Combined sentences into: {output_file}")
            return True
        except Exception as e:
            log.error(f"Error exporting combined chapter audio: {e}")
            return False

    def get_chapter_info(self, chapter_number):
        """Retrieves chapter information by chapter number."""
        for chapter in self.chapter_data:
            if chapter['number'] == chapter_number:
                return chapter
        return None

    def get_html_file_by_chapter(self, chapter_number):
        chapter_info = self.get_chapter_info(chapter_number)
        if chapter_info:
            return chapter_info.get('html_file')
        return None

    def get_all_chapter_files(self):
        """Returns a list of all chapter audio files."""
        return sorted(list(self.chapters_dir.glob(f'*.{default_audio_proc_format}')), key=self._get_chapter_num_from_filepath)

    def _get_chapter_num_from_filepath(self, filepath):
        """Extracts chapter number from filepath (for sorting)."""
        match = re.search(r'chapter_(\d+)', filepath.name)
        if match:
            return int(match.group(1))
        return float('inf')
