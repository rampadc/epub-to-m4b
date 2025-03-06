import os
import json
import zipfile
import uuid
import tempfile
import shutil
from lxml import etree
from ebooklib import epub
from typing import Optional, List, Dict, Any

class MediaOverlayCreator:
    def __init__(self, epub_path: str, sync_data_path: str, output_path: Optional[str] = None):
        """
        Initialize the media overlay creator

        Args:
            epub_path: Path to the EPUB file
            sync_data_path: Path to the synchronization JSON file
            output_path: Where to save the result (default: original filename + "_audio.epub")
        """
        self.epub_path = epub_path
        self.sync_data_path = sync_data_path

        if not output_path:
            base_name = os.path.splitext(epub_path)[0]
            self.output_path = f"{base_name}_audio.epub"
        else:
            self.output_path = output_path

        # Load sync data
        with open(sync_data_path, 'r', encoding='utf-8') as f:
            self.sync_data: Dict[str, Any] = json.load(f)

        self.book: Optional[epub.EpubBook] = None
        self.total_duration: float = 0
        self.temp_dir: Optional[str] = None

    def process(self) -> str:
        """Main processing function that adds media overlay to EPUB"""
        try:
            # Create temp dir for processing
            self.temp_dir = tempfile.mkdtemp()

            # Read EPUB
            self.book = epub.read_epub(self.epub_path)

            # Create a clean copy of the epub book
            self.create_clean_book()

            # First convert any provided audio file to proper fragments
            audio_path = self.prepare_audio()

            # Create Media Overlay documents
            self.create_media_overlays(audio_path)

            # Add metadata for media overlay
            self.add_media_overlay_metadata()

            # Write the final EPUB
            if self.book is not None:  # Add this check to satisfy pyright
                epub.write_epub(self.output_path, self.book)

            print(f"Successfully created EPUB with media overlay: {self.output_path}")
            return self.output_path

        finally:
            # Clean up temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def create_clean_book(self) -> None:
        """Create a clean copy of the book without problematic metadata"""
        if self.book is None:
            raise ValueError("EPUB book not initialized")

        # Create a new book
        clean_book = epub.EpubBook()

        # Copy basic metadata directly (title, language, etc.)
        for dc_key in ['title', 'language', 'creator', 'identifier', 'publisher', 'description']:
            values = self.book.get_metadata('DC', dc_key)
            if values:
                for value_data in values:
                    # Handle both simple string values and (value, attributes) tuples
                    if isinstance(value_data, str):
                        clean_book.add_metadata('DC', dc_key, value_data)
                    elif isinstance(value_data, tuple) and len(value_data) == 2:
                        value, attrs = value_data
                        # Filter out problematic attributes
                        clean_attrs = {}
                        for k, v in attrs.items():
                            if not k.startswith('opf:'):
                                clean_attrs[k] = v
                        clean_book.add_metadata('DC', dc_key, value, clean_attrs)

        # Copy items
        for item_id in self.book.get_items():
            clean_book.add_item(item_id)

        # Copy TOC
        clean_book.toc = self.book.toc

        # Copy spine
        for item_id, linear in self.book.spine:
            clean_book.spine.append((item_id, linear))

        # Set as the new book
        self.book = clean_book

    def format_smil_duration(self, seconds: float) -> str:
        """
        Format duration in valid SMIL clock value format

        SMIL clock values must be in the format: hh:mm:ss.fraction or mm:ss.fraction
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        fraction = int((seconds - int(seconds)) * 1000)

        if hours > 0:
            return f"{hours:01d}:{minutes:02d}:{secs:02d}.{fraction:03d}"
        else:
            return f"{minutes:01d}:{secs:02d}.{fraction:03d}"

    def prepare_audio(self) -> str:
        """Prepare audio files for inclusion in the EPUB"""
        audio_file = self.sync_data.get('audio_file')
        if not audio_file or not os.path.exists(audio_file):
            raise ValueError("No valid audio file found in sync data")

        # Copy audio to temp dir
        audio_filename = os.path.basename(audio_file)
        if self.temp_dir is None:
            raise ValueError("Temporary directory not initialized")

        audio_path = os.path.join(self.temp_dir, audio_filename)
        shutil.copy(audio_file, audio_path)

        # Add audio file to EPUB
        audio_item = epub.EpubItem(
            uid=f"audio_{uuid.uuid4().hex[:8]}",
            file_name=f"Audio/{audio_filename}",
            media_type="audio/mpeg",
            content=open(audio_file, 'rb').read()
        )

        if self.book is None:
            raise ValueError("EPUB book not initialized")

        self.book.add_item(audio_item)

        return audio_path

    def add_media_overlay_metadata(self) -> None:
        """Add metadata for media overlay"""
        if self.book is None:
            raise ValueError("EPUB book not initialized")

        # Add total duration metadata
        total_duration_ms = self.sync_data.get('total_duration_ms', 0)
        total_duration_sec = total_duration_ms / 1000
        self.total_duration = total_duration_sec

        # Format duration in SMIL format using the standardized method
        smil_duration = self.format_smil_duration(total_duration_sec)

        # Add to metadata - use proper namespace and attributes
        self.book.add_metadata('OPF', 'meta', smil_duration, {'property': 'media:duration'})

        # Add active class metadata
        self.book.add_metadata('OPF', 'meta', '-epub-media-overlay-active', {'property': 'media:active-class'})

    def create_media_overlays(self, audio_path: str) -> None:
        """Create SMIL files for media overlay"""
        if self.book is None:
            raise ValueError("EPUB book not initialized")

        # Get book spine to understand reading order
        spine_items = self.book.spine
        audio_filename = os.path.basename(audio_path)

        # Process each chapter
        for chapter_idx, chapter_data in enumerate(self.sync_data.get('chapters', [])):
            chapter_num = chapter_data.get('number')
            chapter_start_ms = chapter_data.get('start_time')
            chapter_end_ms = chapter_data.get('end_time')

            # Get fragments for this chapter
            fragments = [f for f in self.sync_data.get('fragments', [])
                         if f.get('chapter') == chapter_num]

            # Skip if there are no fragments
            if not fragments:
                continue

            # Get spine item for this chapter (if we have enough items)
            if chapter_idx < len(spine_items):
                # Get the item ID from spine
                item_id = spine_items[chapter_idx][0]
                # Get the actual item
                content_item = self.book.get_item_with_id(item_id)

                if content_item:
                    # Process content to add fragment IDs
                    content = content_item.get_content().decode('utf-8')
                    content = self.tag_sentences(content_item.id, content, fragments)
                    content_item.set_content(content.encode('utf-8'))

                    # Create SMIL file
                    smil_id = f"{content_item.id}_overlay"
                    smil_filename = f"MediaOverlays/chapter_{chapter_num}.smil"

                    # Create SMIL content
                    smil_content = self.create_smil_content(
                        content_item, fragments, audio_filename, smil_id)

                    # Add SMIL file to EPUB
                    smil_item = epub.EpubItem(
                        uid=smil_id,
                        file_name=smil_filename,
                        media_type="application/smil+xml",
                        content=smil_content.encode('utf-8')
                    )
                    self.book.add_item(smil_item)

                    # Link SMIL to content item
                    content_item.media_overlay = smil_id

                    # Add chapter duration metadata with proper SMIL clock value
                    chapter_duration_sec = (chapter_end_ms - chapter_start_ms) / 1000
                    chapter_smil_duration = self.format_smil_duration(chapter_duration_sec)

                    self.book.add_metadata('OPF', 'meta', chapter_smil_duration,
                                          {'property': 'media:duration', 'refines': f"#{smil_id}"})

    def tag_sentences(self, chapter_id: str, content: str, fragments: List[Dict[str, Any]]) -> str:
        """Tag sentences in content for media overlay"""
        # Simple implementation to add span tags with ids
        for i, fragment in enumerate(fragments):
            fragment_id = f"{chapter_id}-sentence{i+1}"
            fragment_text = fragment.get('text', '')

            if fragment_text and fragment_text in content:
                # Replace only first occurrence to avoid duplicates
                tagged_text = f'<span id="{fragment_id}">{fragment_text}</span>'
                content = content.replace(fragment_text, tagged_text, 1)

        return content

    def create_smil_content(self, chapter: epub.EpubItem, fragments: List[Dict[str, Any]],
                           audio_filename: str, smil_id: str) -> str:
        """Create SMIL content for media overlay"""
        smil = f"""<?xml version="1.0" encoding="UTF-8"?>
<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">
  <body id="{smil_id}_overlay" epub:textref="../{chapter.file_name}" epub:type="chapter">
    <seq>"""

        for i, fragment in enumerate(fragments):
            fragment_id = f"{chapter.id}-sentence{i+1}"
            start_time_sec = fragment.get('start_time', 0) / 1000  # Convert to seconds
            end_time_sec = fragment.get('end_time', 0) / 1000  # Convert to seconds

            # Format times as valid SMIL clock values
            start_time = self.format_smil_duration(start_time_sec)
            end_time = self.format_smil_duration(end_time_sec)

            smil += f"""
      <par id="{fragment_id}">
        <text src="../{chapter.file_name}#{fragment_id}" />
        <audio src="../Audio/{audio_filename}" clipBegin="{start_time}" clipEnd="{end_time}" />
      </par>"""

        smil += """
    </seq>
  </body>
</smil>"""

        return smil


def add_media_overlay_to_epub(epub_path: str, sync_data_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Add media overlay (audio synchronization) to an existing EPUB file

    Args:
        epub_path: Path to the existing EPUB file
        sync_data_path: Path to the JSON file containing text-audio synchronization data
        output_path: Path for the output EPUB file with media overlay

    Returns:
        Path to the enhanced EPUB file if successful, None otherwise
    """
    try:
        creator = MediaOverlayCreator(epub_path, sync_data_path, output_path)
        return creator.process()
    except Exception as e:
        import traceback
        print(f"Error adding media overlay to EPUB: {e}")
        traceback.print_exc()
        return None
