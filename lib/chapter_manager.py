import os
import re
import subprocess
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from .preprocess import normalize_text, get_sentences
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class ChapterManager:
    def __init__(self, epub_path, output_dir, default_voice='af_heart'):
        self.epub_path = Path(epub_path)
        self.output_dir = Path(output_dir)
        self.chapters_dir = self.output_dir / "chapters"
        # self.sentences_dir = self.chapters_dir / "sentences"  # No longer needed
        self.temp_dir = self.output_dir / "temp"  # For intermediary WAV files
        self.default_voice = default_voice
        self.book = None
        self._load_epub()
        self.chapter_data = []
        self._extract_chapters()

    def _load_epub(self):
        try:
            self.book = epub.read_epub(self.epub_path)
            log.info(f"Successfully read EPUB: {self.epub_path}")
        except Exception as e:
            log.error(f"Error reading EPUB: {e}")
            raise

    def _extract_chapters(self):
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

                title_tag = soup.find(['h1', 'h2', 'h3'])
                chapter_title = title_tag.get_text().strip() if title_tag else f"Chapter {chapter_number}"

                self.chapter_data.append({
                    'number': chapter_number,
                    'title': chapter_title,
                    'sentences': sentences,
                    'audio_files': [],  #  No longer storing individual sentence audio files
                    'html_file': doc.file_name
                })
                chapter_number += 1
            except Exception as e:
                log.error(f"Error processing document {doc.file_name}: {e}")
                continue

    def prepare_directories(self):
        """Creates the necessary output directories."""
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        # self.sentences_dir.mkdir(parents=True, exist_ok=True)  # No longer needed
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Prepared directories: {self.chapters_dir}, {self.temp_dir}")

    def get_chapter_audio_filepath(self, chapter_number):
        """Returns the MP3 file path for a chapter."""
        return self.chapters_dir / f"chapter_{chapter_number}.mp3"

    # def get_sentence_audio_filepath(self, chapter_number, sentence_number): # No longer needed
    #     """Returns the MP3 file path for a sentence."""
    #     return self.sentences_dir / f"chapter_{chapter_number}_sentence_{sentence_number}.mp3"

    def get_temp_wav_filepath(self, chapter_number, sentence_number):
        """Returns the temporary WAV file path for a sentence."""
        return self.temp_dir / f"chapter_{chapter_number}_sentence_{sentence_number}.wav"

    def get_all_sentence_data(self, voice_map=None):
        """Returns a list of all sentences with associated data for processing."""
        all_sentences = []

        for chapter_info in self.chapter_data:
            chapter_num = chapter_info['number']
            voice = voice_map.get(chapter_num, self.default_voice) if voice_map else self.default_voice

            for i, sentence in enumerate(chapter_info['sentences']):
                # mp3_file = self.get_sentence_audio_filepath(chapter_num, i) # No longer needed
                wav_file = self.get_temp_wav_filepath(chapter_num, i)
                all_sentences.append({
                    'chapter': chapter_num,
                    'sentence_num': i,
                    'text': sentence,
                    # 'file': mp3_file,  # No longer needed
                    'wav_file': wav_file,
                    'voice': voice
                })
        return all_sentences

    def wav_to_mp3(self, wav_path, mp3_path):
        """Convert WAV to MP3 using ffmpeg."""
        try:
            subprocess.run([
                'ffmpeg',
                '-i', str(wav_path),
                '-c:a', 'libmp3lame',
                '-q:a', '4',
                '-y',
                str(mp3_path)
            ], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            log.error(f"Error converting WAV to MP3: {e.stderr.decode()}")
            return False

    def combine_sentences_to_chapter(self, chapter_number):
        """Combines sentence WAV files into chapter MP3."""
        chapter_info = self.get_chapter_info(chapter_number)
        if not chapter_info:
            log.error(f"Chapter {chapter_number} not found.")
            return False

        output_file = self.get_chapter_audio_filepath(chapter_number)
        sentence_data = [data for data in self.get_all_sentence_data()
                        if data['chapter'] == chapter_number]

        # Get WAV files for concatenation
        wav_files = [data['wav_file'] for data in sentence_data]
        valid_wav_files = [f for f in wav_files if f.exists()]

        if not valid_wav_files:
            log.warning(f"No WAV files found for chapter {chapter_number}.")
            return False

        try:
            # Create concat list
            concat_list = self.temp_dir / f"concat_list_{chapter_number}.txt"
            with open(concat_list, 'w') as f:
                for wav in valid_wav_files:
                    f.write(f"file '{wav.absolute()}'\n")

            # Concatenate WAV files
            temp_wav = self.temp_dir / f"chapter_{chapter_number}_combined.wav"
            subprocess.run([
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', str(concat_list),
                '-c', 'copy',
                '-y', str(temp_wav)
            ], check=True)

            # Convert to final MP3
            success = self.wav_to_mp3(temp_wav, output_file)

            # Clean up temporary files
            temp_wav.unlink(missing_ok=True)
            concat_list.unlink(missing_ok=True)

            if success:
                log.info(f"Successfully created chapter {chapter_number} audio")
                return output_file # Return the path to the created MP3
            else:
                log.error(f"Failed to convert chapter {chapter_number} to MP3")
                return False

        except Exception as e:
            log.error(f"Error combining audio for chapter {chapter_number}: {e}")
            return False

    def get_chapter_info(self, chapter_number):
        """Retrieves chapter information by chapter number."""
        for chapter in self.chapter_data:
            if chapter['number'] == chapter_number:
                return chapter
        return None

    def get_html_file_by_chapter(self, chapter_number):
        """Gets the HTML file associated with a chapter number."""
        chapter_info = self.get_chapter_info(chapter_number)
        if chapter_info:
            return chapter_info.get('html_file')
        return None

    def get_all_chapter_files(self):
        """Returns a sorted list of all chapter audio files (MP3s)."""
        return sorted(list(self.chapters_dir.glob('chapter_*.mp3')),
                     key=self._get_chapter_num_from_filepath)

    def _get_chapter_num_from_filepath(self, filepath):
        """Extracts chapter number from filepath (for sorting)."""
        match = re.search(r'chapter_(\d+)', filepath.name)
        if match:
            return int(match.group(1))
        return float('inf')

    def cleanup_temp_files(self):
        """Clean up all temporary WAV files."""
        try:
            for wav_file in self.temp_dir.glob('*.wav'):
                wav_file.unlink(missing_ok=True)
            for txt_file in self.temp_dir.glob('*.txt'):
                txt_file.unlink(missing_ok=True)
            self.temp_dir.rmdir()
            log.info("Successfully cleaned up temporary files")
        except Exception as e:
            log.error(f"Error cleaning up temporary files: {e}")

    # def create_sentence_mp3(self, wav_path, mp3_path): # No longer needed
    #    """Create MP3 file from WAV for media overlay."""
    #    return self.wav_to_mp3(wav_path, mp3_path)
