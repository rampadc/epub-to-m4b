import os
import re
import shutil
import argparse
import sys
from ebooklib import epub
from pydub import AudioSegment
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import zipfile

from config import DEFAULT_API_ENDPOINT, DEFAULT_API_KEY, DEFAULT_MAX_WORKERS, DEFAULT_MODEL, default_audio_proc_format
from lib.assemble_audio import assemble_audiobook_m4b, build_text_audio_mapping, combine_audio_sentences
from lib.media_overlay import add_media_overlay_to_epub
from lib.preprocess import DependencyError, convert_to_epub, get_book_metadata, get_chapters, get_cover, prepare_dirs
from lib.process import process_sentence

def get_chapter_num(filename):
    match = re.search(r'chapter_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return float('inf')

def read_epub_safely(epub_path):
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
                    print(f"Adding missing {css_file} to EPUB file...")
                    epub_zip.writestr(css_file, '/* Empty CSS file */')
    except Exception as e:
        print(f"Warning: Could not preemptively fix EPUB file: {e}")
        # Continue anyway, we'll try to read the file as is

    # Now try to read the EPUB
    try:
        return epub.read_epub(epub_path, {'ignore_ncx': True})
    except KeyError as e:
        # Extract the missing file name from the error
        import re
        missing_file_match = re.search(r"'([^']*)'", str(e))
        if missing_file_match:
            missing_file = missing_file_match.group(1)
            print(f"Warning: EPUB file is missing '{missing_file}'. Attempting to fix...")

            try:
                # Try to add the specific missing file
                with zipfile.ZipFile(epub_path, 'a') as epub_zip:
                    epub_zip.writestr(missing_file, '/* Empty file */')
                print(f"Added missing file: {missing_file}. Trying to read EPUB again...")
                return epub.read_epub(epub_path, {'ignore_ncx': True})
            except Exception as fix_error:
                print(f"Error while fixing EPUB: {fix_error}")
                # If we still can't read it, try a more drastic approach
                try:
                    print("Attempting to extract and repackage the EPUB...")
                    import tempfile
                    import os
                    import shutil

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
                        print("EPUB file has been repackaged. Trying to read again...")

                        return epub.read_epub(epub_path, {'ignore_ncx': True})
                    finally:
                        # Clean up the temporary directory
                        shutil.rmtree(temp_dir)
                except Exception as repackage_error:
                    print(f"Error while repackaging EPUB: {repackage_error}")
                    raise e
        else:
            # For other KeyError issues, just raise the original error
            raise
    except Exception as e:
        print(f"Error reading EPUB file: {e}")
        raise


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
    parser.add_argument("--format", default="m4b", choices=["mp3", "m4b", "wav", "ogg", "flac"],
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
    parser.add_argument("--create-media-overlay", action="store_true",
                        help="Create EPUB with media overlay (synchronized text and audio)")

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
        epub_book = read_epub_safely(dirs["epub_path"])

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

        # Generate text-audio synchronization data
        print("\nBuilding text-audio synchronization data...")
        mapping_file = build_text_audio_mapping(chapters, all_sentences_data, dirs)

        if not mapping_file:
            raise Exception("Failed to create text-audio synchronization mapping")

        output_base = os.path.splitext(os.path.basename(args.input))[0]

        # Determine output mode based on media overlay setting
        if args.create_media_overlay:
            # Media overlay mode - only create EPUB with embedded audio
            print("\nMedia overlay mode: Creating EPUB with synchronized audio...")

            # First create a temporary MP3 file for embedding in the EPUB
            temp_mp3_file = os.path.join(dirs["process_dir"], "temp_audiobook.mp3")

            # Combine all chapter files into an MP3
            combined = AudioSegment.empty()

            # Sort chapter files by chapter number


            chapter_files.sort(key=get_chapter_num)

            print(f"Combining {len(chapter_files)} chapters for EPUB audio...")
            for chapter_file in tqdm(chapter_files):
                if os.path.exists(chapter_file):
                    chapter_audio = AudioSegment.from_file(chapter_file, format=default_audio_proc_format)
                    combined += chapter_audio

            # Export as MP3 for embedding in EPUB
            combined.export(temp_mp3_file, format="mp3", bitrate="128k")
            print(f"Created temporary MP3 file for EPUB: {temp_mp3_file}")

            # Update mapping file with the MP3 file path
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    sync_data = json.load(f)

                sync_data["audio_file"] = temp_mp3_file

                with open(mapping_file, 'w', encoding='utf-8') as f:
                    json.dump(sync_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Could not update mapping file with audio path: {e}")

            # Create EPUB with media overlay
            overlay_path = add_media_overlay_to_epub(
                dirs["epub_path"],
                mapping_file,
                output_path=os.path.join(dirs["process_dir"], f"{output_base}_audio.epub")
            )

            if overlay_path:
                print(f"\nCreated EPUB with synchronized audio: {overlay_path}")
                # Attempt to clean up temporary MP3 file
                try:
                    os.remove(temp_mp3_file)
                except:
                    pass
                return 0
            else:
                print("\nFailed to create EPUB with media overlay")
                return 1

        else:
            # Standard mode - create standalone audiobook
            output_file = os.path.join(dirs["process_dir"], f"{output_base}.{args.format}")

            # Assemble final audiobook based on format
            if args.format.lower() == "m4b":
                # Assemble as M4B with chapter markers
                print("\nAssembling final audiobook with chapters (M4B format)...")
                final_audiobook = assemble_audiobook_m4b(
                    dirs["chapters_dir"],
                    output_file,
                    metadata,
                    cover_file,
                    mapping_file
                )
            else:
                # Assemble as MP3 or other format using PyDub
                print(f"\nAssembling final audiobook ({args.format.upper()} format)...")
                # Ensure output file has correct extension
                output_file = f"{os.path.splitext(output_file)[0]}.{args.format}"

                # Combine all chapter files
                combined = AudioSegment.empty()

                chapter_files.sort(key=get_chapter_num)

                print(f"Combining {len(chapter_files)} chapters...")
                for chapter_file in tqdm(chapter_files):
                    if os.path.exists(chapter_file):
                        chapter_audio = AudioSegment.from_file(chapter_file, format=default_audio_proc_format)
                        combined += chapter_audio

                # Set bitrate based on format
                bitrate = "128k"  # Default for MP3
                if args.format == "flac":
                    # FLAC is lossless so bitrate isn't specified the same way
                    combined.export(output_file, format=args.format)
                else:
                    combined.export(output_file, format=args.format, bitrate=bitrate)

                # Copy mapping file alongside the output
                if mapping_file and os.path.exists(mapping_file):
                    mapping_dest = f"{os.path.splitext(output_file)[0]}_sync.json"
                    shutil.copy(mapping_file, mapping_dest)
                    print(f"Copied synchronization data to: {mapping_dest}")

                final_audiobook = output_file

            if final_audiobook and os.path.exists(final_audiobook):
                print(f"\nAudiobook created successfully: {final_audiobook}")
                print(f"Text-audio synchronization data: {mapping_file}")
                return 0
            else:
                print("\nFailed to create audiobook")
                return 1

    except DependencyError:
        # Already handled in the exception
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
