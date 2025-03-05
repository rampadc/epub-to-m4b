import json
import os
import shutil
import subprocess
import traceback
import re
from pydub.audio_segment import AudioSegment
from tqdm import tqdm
from config import default_audio_proc_format


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
