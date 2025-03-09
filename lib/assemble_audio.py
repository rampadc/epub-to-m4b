import json
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

from pydub.audio_segment import AudioSegment
from tqdm import tqdm

import logging

log = logging.getLogger(__name__)


def get_chapter_num(filename):
    """Extracts chapter number from filename."""
    match = re.search(r'chapter_(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else float('inf')

def build_text_audio_mapping(chapter_manager, sentence_data):
    """Builds a mapping of text fragments to audio timestamps (using chapter MP3s)."""
    try:
        mapping = {
            "version": "1.0",
            "book_title": "Unknown",  #  Get this from metadata if available
            "fragments": [],
            "chapters": []
        }

        current_time_ms = 0

        log.info("Calculating audio timings for synchronization...")

        # Iterate through chapters, not sentences.
        chapter_files = chapter_manager.get_all_chapter_files() # Get the chapter MP3s.
        for chapter_file in chapter_files:
            chapter_num = get_chapter_num(chapter_file)  # Extract from filename
            chapter_info = chapter_manager.get_chapter_info(chapter_num)

            if not chapter_file.exists() or not chapter_info:
                log.warning(f"Skipping chapter {chapter_num} - File or info missing.")
                continue

            try:
                audio = AudioSegment.from_file(chapter_file, format='mp3')
                chapter_duration_ms = len(audio)

                mapping["chapters"].append({
                    "number": chapter_num,
                    "title": chapter_info['title'],
                    "start_time": current_time_ms,
                    "end_time": current_time_ms + chapter_duration_ms,
                    "fragment_count": len(chapter_info['sentences'])  # Count of sentences
                })

                #  Now, create fragment entries *within* this chapter.
                sentence_start_time = current_time_ms
                for i, sentence in enumerate(chapter_info['sentences']):
                    #  We have to *estimate* sentence duration within the chapter.
                    #  A more accurate method requires analyzing *every* WAV, which is slow.
                    estimated_sentence_duration = chapter_duration_ms / len(chapter_info['sentences'])

                    fragment = {
                        "chapter": chapter_num,
                        "text": sentence,
                        "start_time": sentence_start_time,
                        "end_time": sentence_start_time + estimated_sentence_duration,
                        "audio_file": str(chapter_file)  #  Reference the *chapter* MP3
                    }
                    mapping["fragments"].append(fragment)
                    sentence_start_time += estimated_sentence_duration  # Increment for next sentence

                current_time_ms += chapter_duration_ms  #  Advance to the *next* chapter.

            except Exception as e:
                log.error(f"Error processing timing for chapter {chapter_num}: {e}")
                # Don't raise here; continue to the next chapter
                traceback.print_exc()

        mapping["total_duration_ms"] = current_time_ms

        mapping_file = chapter_manager.output_dir / "text_audio_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        log.info(f"Created text-audio mapping: {mapping_file}")
        return mapping_file

    except Exception as e:
        log.error(f"Error building text-audio mapping: {e}")
        traceback.print_exc()  #  Print the traceback
        return None


def assemble_audiobook_m4b(chapter_manager, output_file, metadata, cover_file, mapping_file=None):
    """Assembles the final audiobook in M4B format."""
    try:
        if not shutil.which('ffmpeg'):
            log.error("FFmpeg is required for M4B creation.")
            return None

        chapter_files = chapter_manager.get_all_chapter_files()
        if not chapter_files:
            log.error("No chapter files found!")
            return None

        log.info(f"Processing {len(chapter_files)} chapters...")

        # Verify files before concatenation
        valid_chapter_files = []
        skipped_files = []
        for chapter_file in chapter_files:
            # Check if file is valid using ffprobe
            verify_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(chapter_file)
            ]
            try:
                subprocess.run(verify_cmd, check=True, capture_output=True, text=True)
                valid_chapter_files.append(chapter_file)
            except subprocess.CalledProcessError:
                skipped_files.append(chapter_file)
                log.warning(f"Skipping corrupted file: {chapter_file}")

        if skipped_files:
            log.warning(f"Skipped {len(skipped_files)} corrupted files")
            skipped_files_log = chapter_manager.output_dir / "skipped_files.txt"
            with open(skipped_files_log, 'w', encoding='utf-8') as f:
                f.write(f"Skipped files ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n")
                for file in skipped_files:
                    f.write(f"{file}\n")
            log.warning(f"List of skipped files written to: {skipped_files_log}")

        if not valid_chapter_files:
            log.error("No valid audio files found!")
            return None

        # Create a temporary file list for FFmpeg concatenation
        concat_list = chapter_manager.output_dir / "concat_list.txt"
        with open(concat_list, 'w', encoding='utf-8') as f:
            for chapter_file in valid_chapter_files:
                f.write(f"file '{chapter_file.absolute()}'\n")

        # Use FFmpeg to directly concatenate the files
        combined_audio_file = chapter_manager.output_dir / "combined_audio.mp3"
        concat_cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            '-y',
            str(combined_audio_file)
        ]

        log.info("Combining audio files using FFmpeg...")
        try:
            subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            log.error(f"Error combining audio files: {e.stderr}")
            return None

        # Calculate chapter timestamps using ffprobe
        chapter_timestamps = []
        current_pos_ms = 0

        for idx, chapter_file in enumerate(valid_chapter_files):
            try:
                duration_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(chapter_file)
                ]
                duration_output = subprocess.check_output(duration_cmd, text=True)
                duration_ms = int(float(duration_output.strip()) * 1000)

                chapter_timestamps.append({
                    'number': idx + 1,
                    'start_ms': current_pos_ms,
                    'end_ms': current_pos_ms + duration_ms  # Corrected end_ms calculation
                })
                current_pos_ms += duration_ms  # Increment current position
                log.info(f"Added Chapter {idx + 1} ({duration_ms / 1000:.2f} seconds)")
            except (subprocess.CalledProcessError, ValueError) as e:
                log.error(f"Error processing chapter {idx + 1}: {e}")
                continue  # Continue to the next chapter even if one fails


        metadata_file = chapter_manager.output_dir / "metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(";FFMETADATA1\n")
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

            if metadata.get('date'):
                year_match = re.search(r'\b(19|20)\d{2}\b', str(metadata['date']))
                year = year_match.group(0) if year_match else str(datetime.now().year)
                f.write(f"year={year}\n")

            if metadata.get('identifiers'):
                if metadata['identifiers'].get('isbn'):
                    f.write(f"isbn={metadata['identifiers']['isbn']}\n")
                if metadata['identifiers'].get('mobi-asin'):
                    f.write(f"asin={metadata['identifiers']['mobi-asin']}\n")

            if mapping_file and mapping_file.exists():
                f.write(f"comment=This audiobook includes text synchronization data: {mapping_file.name}\n")

            for chapter in chapter_timestamps:
                f.write("\n[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={chapter['start_ms']}\n")
                f.write(f"END={chapter['end_ms']}\n")
                f.write(f"title=Chapter {chapter['number']}\n")


        output_file = Path(output_file)
        if output_file.suffix.lower() != '.m4b':
            output_file = output_file.with_suffix('.m4b')

        ffmpeg_cmd = [
            'ffmpeg',
            '-i', str(combined_audio_file),
            '-i', str(metadata_file)
        ]

        if cover_file and Path(cover_file).exists():
            ffmpeg_cmd.extend(['-i', str(cover_file)])
            ffmpeg_cmd.extend([
                '-map', '0:a',
                '-map', '2:v',  # Map the cover image
                '-disposition:v', 'attached_pic'
            ])
        else:
            ffmpeg_cmd.extend(['-map', '0:a']) # Map only audio if no cover


        ffmpeg_cmd.extend([
            '-c:a', 'aac',  # Use AAC for M4B
            '-b:a', '128k', # Set bitrate
            '-ar', '44100',
            '-movflags', '+faststart', # For streaming
            '-map_metadata', '1',   # Copy metadata from the metadata file
            '-y',  # Overwrite output file if it exists
            str(output_file)
        ])

        log.info(f"Creating M4B with FFmpeg: {' '.join(ffmpeg_cmd)}")
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if process.stdout:
             for line in process.stdout:
                if 'size=' in line or 'time=' in line or 'speed=' in line:  # Progress info
                    print(f"\rFFmpeg: {line.strip()}", end='')
        process.wait()
        print() # Newline after progress


        if process.returncode == 0:
            log.info(f"Successfully created M4B: {output_file}")
            if mapping_file and mapping_file.exists():
                mapping_dest = output_file.with_name(f"{output_file.stem}_sync.json")
                shutil.copy(mapping_file, mapping_dest)
                log.info(f"Copied sync data to: {mapping_dest}")
            combined_audio_file.unlink()  # Delete the combined MP3
            metadata_file.unlink()       # Delete the metadata file
            concat_list.unlink()         # Clean up the concat list file
            return output_file
        else:
            log.error(f"Error creating M4B. FFmpeg exited with code {process.returncode}")
            return None

    except Exception as e:
        log.error(f"Error assembling M4B: {e}")
        traceback.print_exc()  #  Print the full traceback
        return None
