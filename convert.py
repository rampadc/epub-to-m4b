import os
import argparse
import sys
import shutil
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from config import DEFAULT_MAX_WORKERS
from lib.preprocess import DependencyError, convert_to_epub, get_book_metadata, get_cover, prepare_dirs
from lib.chapter_manager import ChapterManager
from lib.assemble_audio import assemble_audiobook_m4b, build_text_audio_mapping  # Corrected import
from lib.process import process_sentence
from lib.tts import get_voices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("epub2audiobook")

def read_epub_safely(epub_path):
    """
    Safely read an EPUB file, attempting to fix common issues if needed.
    """
    from ebooklib import epub
    import zipfile
    import re

    common_missing_files = [
        'page_styles.css',
        'stylesheet.css',
        'style.css',
        'styles.css'
    ]

    # First, proactively add all potentially missing files
    try:
        with zipfile.ZipFile(epub_path, 'a') as epub_zip:
            existing_files = set(epub_zip.namelist())

            for css_file in common_missing_files:
                if css_file not in existing_files:
                    log.info(f"Adding missing {css_file} to EPUB file...")
                    epub_zip.writestr(css_file, '/* Empty CSS file */')
    except Exception as e:
        log.warning(f"Could not preemptively fix EPUB file: {e}")
        # Continue anyway, we'll try to read the file as is

    # Now try to read the EPUB
    try:
        return epub.read_epub(epub_path, {'ignore_ncx': True})
    except KeyError as e:
        # Extract the missing file name from the error
        missing_file_match = re.search(r"'([^']*)'", str(e))
        if missing_file_match:
            missing_file = missing_file_match.group(1)
            log.warning(f"EPUB file is missing '{missing_file}'. Attempting to fix...")

            try:
                # Try to add the specific missing file
                with zipfile.ZipFile(epub_path, 'a') as epub_zip:
                    epub_zip.writestr(missing_file, '/* Empty file */')
                log.info(f"Added missing file: {missing_file}. Trying to read EPUB again...")
                return epub.read_epub(epub_path, {'ignore_ncx': True})
            except Exception as fix_error:
                log.error(f"Error while fixing EPUB: {fix_error}")
                # If we still can't read it, try a more drastic approach
                try:
                    log.info("Attempting to extract and repackage the EPUB...")
                    import tempfile

                    # Create a temporary directory
                    temp_dir = tempfile.mkdtemp()
                    try:
                        # Extract the EPUB
                        with zipfile.ZipFile(epub_path, 'r') as epub_zip:
                            epub_zip.extractall(temp_dir)

                        # Create all missing CSS files
                        for css_file in common_missing_files:
                            css_path = os.path.join(temp_dir, css_file)
                            if not os.path.exists(css_path):
                                with open(css_path, 'w') as f:
                                    f.write('/* Empty CSS file */')

                        # Create a new EPUB file
                        temp_epub = epub_path + '.fixed'
                        with zipfile.ZipFile(temp_epub, 'w') as new_epub:
                            for root, _, files in os.walk(temp_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, temp_dir)
                                    new_epub.write(file_path, arcname)

                        # Replace the original with the fixed version
                        shutil.move(temp_epub, epub_path)
                        log.info("EPUB file has been repackaged. Trying to read again...")

                        return epub.read_epub(epub_path, {'ignore_ncx': True})
                    finally:
                        # Clean up the temporary directory
                        shutil.rmtree(temp_dir)
                except Exception as repackage_error:
                    log.error(f"Error while repackaging EPUB: {repackage_error}")
                    raise e
        else:
            # For other KeyError issues, just raise the original error
            raise
    except Exception as e:
        log.error(f"Error reading EPUB file: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert eBooks to audiobooks using Kokoro TTS")
    parser.add_argument("--input", "-i", required=True, help="Input eBook file")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Maximum number of concurrent audio processing tasks (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--format", default="m4b", choices=["mp3", "m4b", "wav", "ogg", "flac"],
                        help="Output format (default: m4b)")
    parser.add_argument("--sync-text", action="store_true",
                        help="Generate text-audio synchronization data for text highlighting")
    parser.add_argument("--chapter-start", type=int, default=1,
                        help="Starting chapter number to process (default: 1)")
    parser.add_argument("--chapter-end", type=int, default=None,
                        help="Ending chapter number to process (inclusive, default: process all chapters)")
    parser.add_argument("--voice", default="af_heart", choices=get_voices(),
                        help="Kokoro voice to use")
    parser.add_argument("--voice-map", help="JSON file mapping chapter numbers to specific voices")

    args = parser.parse_args()

    try:
        # Prepare directories
        dirs = prepare_dirs(args.input, args.output_dir)
        process_dir = Path(dirs["process_dir"])

        # Convert to EPUB if needed
        if not args.input.lower().endswith('.epub'):
            log.info("Converting input file to EPUB format...")
            convert_to_epub(dirs["ebook"], dirs["epub_path"])
        else:
            # If input is already EPUB, just copy it
            input_path = Path(args.input).absolute()
            epub_path = Path(dirs["epub_path"]).absolute()

            if input_path != epub_path:
                shutil.copy(input_path, epub_path)

            # Always use the file in the processing directory
            dirs["epub_path"] = str(epub_path)

        # Read the EPUB
        log.info("Reading EPUB content...")
        epub_book = read_epub_safely(dirs["epub_path"])

        # Extract metadata and cover
        log.info("Extracting book metadata...")
        metadata = get_book_metadata(epub_book)
        cover_file = get_cover(epub_book, dirs["process_dir"])

        if cover_file:
            log.info(f"Extracted cover: {cover_file}")
        else:
            log.warning("No cover image found in the book")

        # Initialize the ChapterManager
        log.info("Initializing chapter manager...")
        chapter_manager = ChapterManager(dirs["epub_path"], process_dir, args.voice)
        chapter_manager.prepare_directories()

        # Load voice map if provided
        voice_map = {}
        if args.voice_map and os.path.exists(args.voice_map):
            try:
                with open(args.voice_map, 'r') as f:
                    voice_map = json.load(f)
                log.info(f"Loaded voice map from {args.voice_map}")
            except Exception as e:
                log.error(f"Error loading voice map: {e}")

        # Filter chapters based on start/end arguments if specified
        chapter_count = len(chapter_manager.chapter_data)
        if args.chapter_start > 1 or args.chapter_end is not None:
            start_chapter = args.chapter_start
            end_chapter = args.chapter_end if args.chapter_end is not None else chapter_count

            # Validate chapter range
            if start_chapter < 1 or start_chapter > chapter_count:
                log.error(f"Starting chapter {start_chapter} is out of range (book has {chapter_count} chapters)")
                return 1

            if end_chapter > chapter_count:
                log.warning(f"Ending chapter {end_chapter} exceeds available chapters. Using last chapter ({chapter_count}) instead.")
                end_chapter = chapter_count

            log.info(f"Processing chapters {start_chapter} to {end_chapter} (out of {chapter_count} total chapters)")

            # Filter chapter_data to only include the specified range
            chapter_manager.chapter_data = [ch for ch in chapter_manager.chapter_data
                                          if start_chapter <= ch['number'] <= end_chapter]
        else:
            log.info(f"Processing all {chapter_count} chapters")

        if not chapter_manager.chapter_data:
            raise DependencyError("Failed to extract any chapters from the eBook")

        # Get all sentence data for processing
        all_sentence_data = chapter_manager.get_all_sentence_data(voice_map)

        # Process all sentences using thread pool
        log.info(f"Processing {len(all_sentence_data)} sentences with {args.max_workers} workers...")
        with tqdm(total=len(all_sentence_data), desc="Processing sentences", unit="sentence") as progress_bar:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []

                for sentence_info in all_sentence_data:
                    #  Check for existing WAV file, not MP3
                    if sentence_info['wav_file'].exists():
                        progress_bar.update(1)
                        continue

                    future = executor.submit(process_sentence, sentence_info)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        #  Result is now (sentence_num, wav_file, success)
                        if result and not result[2]:  # Check if processing was successful
                            log.warning(f"Failed to process sentence {result[0]}")
                    except Exception as exc:
                        log.error(f"Error processing sentence: {exc}")
                    progress_bar.update(1)

        # Combine sentences into chapters and create chapter MP3s
        log.info("Combining sentences into chapters and creating chapter MP3s...")
        for chapter_info in tqdm(chapter_manager.chapter_data, desc="Creating chapters"):
            chapter_num = chapter_info['number']
            chapter_manager.combine_sentences_to_chapter(chapter_num) # This now creates chapter MP3s

        # Get a list of *all* sentence data again, but only for mapping.
        # This is necessary because the chapter combining process modifies the audio file paths.
        all_sentence_data_for_mapping = chapter_manager.get_all_sentence_data(voice_map)

        # Generate text-audio mapping for synchronization
        log.info("Building text-audio synchronization data...")
        # Use the updated all_sentence_data_for_mapping
        mapping_file = build_text_audio_mapping(chapter_manager, all_sentence_data_for_mapping)

        if not mapping_file:
            log.error("Failed to create text-audio synchronization mapping")
            #  Don't exit here; we can still create the audiobook without sync.
            if args.sync_text:
                return 1  # Exit if sync was explicitly requested

        output_base = Path(args.input).stem
        output_file = process_dir / f"{output_base}.{args.format}"

        # Standard mode - create standalone audiobook
        log.info(f"Creating final audiobook in {args.format.upper()} format...")

        final_audiobook = assemble_audiobook_m4b(
            chapter_manager,
            output_file,
            metadata,
            cover_file,
            mapping_file if args.sync_text else None
        )

        if final_audiobook and os.path.exists(final_audiobook):
            log.info(f"Audiobook created successfully: {final_audiobook}")
            if args.sync_text and mapping_file:
                log.info("Text-audio synchronization data included with the audiobook")
            return 0  # Success
        else:
            log.error("Failed to create audiobook")
            return 1 # Fail

    except DependencyError as e:
        # Already handled in the exception
        log.error(f"Dependency error: {str(e)}")
        return 1
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
