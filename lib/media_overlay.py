import traceback


def add_media_overlay_to_epub(epub_path, sync_data_path, output_path=None):
    """
    Add media overlay (audio synchronization) to an existing EPUB file

    Args:
        epub_path: Path to the existing EPUB file
        sync_data_path: Path to the JSON file containing text-audio synchronization data
        output_path: Path for the output EPUB file with media overlay (default: original filename + "_audio.epub")

    Returns:
        Path to the enhanced EPUB file if successful, None otherwise
    """
    try:
        import uuid
        import shutil
        import tempfile
        import zipfile
        import json
        import lxml.etree as etree
        import os.path

        # Register namespaces properly
        etree.register_namespace('epub', 'http://www.idpf.org/2007/ops')
        etree.register_namespace('smil', 'http://www.w3.org/ns/SMIL')

        # Create output path if not specified
        if not output_path:
            base_name = os.path.splitext(epub_path)[0]
            output_path = f"{base_name}_audio.epub"

        # Load synchronization data
        with open(sync_data_path, 'r', encoding='utf-8') as f:
            sync_data = json.load(f)

        # Ensure we have an audio file reference
        if "audio_file" not in sync_data or not os.path.exists(sync_data["audio_file"]):
            print(f"Error: No valid audio file found in sync data. Please ensure the audio file exists.")
            return None

        audio_file = sync_data["audio_file"]
        audio_filename = os.path.basename(audio_file)

        # Create temporary directory for working with EPUB contents
        temp_dir = tempfile.mkdtemp()

        # Extract EPUB
        print(f"Extracting EPUB to: {temp_dir}")
        with zipfile.ZipFile(epub_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find package file (container.xml points to it)
        container_path = os.path.join(temp_dir, "META-INF", "container.xml")
        container_tree = etree.parse(container_path)
        package_path_rel = container_tree.xpath(
            "//*[local-name()='rootfile']/@full-path")[0]
        package_path = os.path.join(temp_dir, package_path_rel)
        package_dir = os.path.dirname(package_path)

        # Parse package file
        package_tree = etree.parse(package_path)
        ns = {'opf': 'http://www.idpf.org/2007/opf',
              'dc': 'http://purl.org/dc/elements/1.1/'}

        # Get spine, manifest, and metadata elements
        spine = package_tree.xpath("//opf:spine", namespaces=ns)[0]
        manifest = package_tree.xpath("//opf:manifest", namespaces=ns)[0]
        metadata = package_tree.xpath("//opf:metadata", namespaces=ns)[0]

        # Get all items from manifest and their corresponding content files
        items = {}
        for item in manifest.xpath("//opf:item", namespaces=ns):
            item_id = item.get('id')
            item_href = item.get('href')
            items[item_id] = item_href

        # Get reading order from spine
        reading_order = []
        for itemref in spine.xpath("//opf:itemref", namespaces=ns):
            idref = itemref.get('idref')
            if idref in items:
                reading_order.append(idref)

        # Create SMIL directory if needed
        smil_dir = os.path.join(package_dir, "smil")
        os.makedirs(smil_dir, exist_ok=True)

        # Create audio directory and copy the audio file
        audio_dir = os.path.join(package_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_dest_path = os.path.join(audio_dir, audio_filename)
        shutil.copy(audio_file, audio_dest_path)
        print(f"Copied audio file: {audio_file} -> {audio_dest_path}")

        # Add audio file to manifest
        audio_id = "audio-file"
        existing_audio = package_tree.xpath(f"//opf:item[@id='{audio_id}']", namespaces=ns)

        if existing_audio:
            # Update existing audio item
            existing_audio[0].set("href", f"audio/{audio_filename}")
            existing_audio[0].set("media-type", "audio/mpeg")
        else:
            # Add new audio item
            audio_element = etree.SubElement(manifest, "{http://www.idpf.org/2007/opf}item")
            audio_element.set("id", audio_id)
            audio_element.set("href", f"audio/{audio_filename}")
            audio_element.set("media-type", "audio/mpeg")

        # Calculate total duration for the entire publication
        total_duration_ms = sync_data.get("total_duration_ms", 0)
        total_duration_sec = total_duration_ms / 1000

        # Add media:duration for entire publication
        existing_duration = metadata.xpath("//opf:meta[@property='media:duration' and not(@refines)]", namespaces=ns)
        if existing_duration:
            existing_duration[0].text = f"{total_duration_sec:.3f}s"
        else:
            duration_element = etree.SubElement(metadata, "{http://www.idpf.org/2007/opf}meta")
            duration_element.set("property", "media:duration")
            duration_element.text = f"{total_duration_sec:.3f}s"

        # Add media:active-class if not already present
        existing_active_class = metadata.xpath("//opf:meta[@property='media:active-class']", namespaces=ns)
        if not existing_active_class:
            active_class = etree.SubElement(metadata, "{http://www.idpf.org/2007/opf}meta")
            active_class.set("property", "media:active-class")
            active_class.text = "reading"

        # Add media:narrator if not already present
        existing_narrator = metadata.xpath("//opf:meta[@property='media:narrator']", namespaces=ns)
        if not existing_narrator:
            narrator = etree.SubElement(metadata, "{http://www.idpf.org/2007/opf}meta")
            narrator.set("property", "media:narrator")
            narrator.text = "TTS Narrator"

        # Store all SMIL durations to ensure they match total duration
        smil_durations = []

        # First pass: pre-process content files to add fragment identifiers
        print("Pre-processing content files to add fragment identifiers...")
        fragment_to_content_map = {}  # Maps fragment ids to (content_file, element_id)

        for chapter_num, chapter_data in enumerate(sync_data.get("chapters", [])):
            # Get fragments for this chapter
            chapter_fragments = [f for f in sync_data.get("fragments", [])
                                if f.get("chapter") == chapter_data.get("number")]

            # Determine content file for this chapter
            if chapter_num < len(reading_order):
                content_id = reading_order[chapter_num]
            else:
                # If we have more chapters than reading order items, use the last one
                content_id = reading_order[-1]

            content_path_rel = items[content_id]
            content_path = os.path.join(package_dir, content_path_rel)

            if os.path.exists(content_path):
                try:
                    # Parse content file
                    parser = etree.XMLParser(ns_clean=True, recover=True)
                    content_tree = etree.parse(content_path, parser)

                    # Find body element
                    body_elem = content_tree.xpath("//body")
                    if not body_elem:
                        print(f"Warning: No body element found in {content_path}")
                        continue

                    body_elem = body_elem[0]

                    # Create a div to hold all our fragments if it doesn't exist
                    fragment_container = body_elem.xpath(".//div[@id='media-overlay-fragments']")
                    if not fragment_container:
                        fragment_container = etree.SubElement(body_elem, "div")
                        fragment_container.set("id", "media-overlay-fragments")
                        fragment_container.set("style", "display:none;")
                    else:
                        fragment_container = fragment_container[0]

                    # Add span elements for each fragment
                    for i, fragment in enumerate(chapter_fragments):
                        frag_id = f"frag-{chapter_num+1}-{i+1}"
                        frag_text = fragment.get("text", "")

                        # Create span with the fragment id
                        span = etree.SubElement(fragment_container, "span")
                        span.set("id", frag_id)
                        span.set("class", "mo-fragment")
                        span.text = frag_text

                        # Store mapping
                        fragment_to_content_map[frag_id] = (content_path_rel, frag_id)

                    # Save modified content file
                    with open(content_path, 'wb') as f:
                        # Add xml declaration and doctype
                        xml_declaration = b'<?xml version="1.0" encoding="utf-8"?>\n'
                        doctype = b'<!DOCTYPE html>\n'
                        f.write(xml_declaration)
                        f.write(doctype)
                        f.write(etree.tostring(content_tree.getroot(),
                                            pretty_print=True,
                                            xml_declaration=False,
                                            encoding="utf-8"))

                except Exception as e:
                    print(f"Error pre-processing content file {content_path}: {e}")
                    traceback.print_exc()

        # Process each chapter in the synchronization data to create SMIL files
        print("Creating SMIL files for each chapter...")
        for chapter_num, chapter_data in enumerate(sync_data.get("chapters", [])):
            chapter_start = chapter_data.get("start_time", 0)
            chapter_end = chapter_data.get("end_time", 0)
            chapter_duration_ms = chapter_end - chapter_start
            chapter_duration_sec = chapter_duration_ms / 1000

            # Store for validation
            smil_durations.append(chapter_duration_sec)

            # Get fragments for this chapter
            chapter_fragments = [f for f in sync_data.get("fragments", [])
                                if f.get("chapter") == chapter_data.get("number")]

            if not chapter_fragments:
                print(f"Warning: No fragments found for chapter {chapter_data.get('number')}")
                continue

            # Determine which content file this chapter belongs to
            if chapter_num < len(reading_order):
                content_id = reading_order[chapter_num]
            else:
                # If we have more chapters than reading order items, use the last one
                content_id = reading_order[-1]

            content_path_rel = items[content_id]

            # Create SMIL file for this chapter
            smil_id = f"smil-{chapter_num+1}"
            smil_filename = f"chapter_{chapter_num+1}.smil"
            smil_path = os.path.join(smil_dir, smil_filename)

            # Create basic SMIL file structure
            smil_root = etree.Element("smil",
                                   xmlns="http://www.w3.org/ns/SMIL",
                                   version="3.0")
            smil_head = etree.SubElement(smil_root, "head")
            metadata_elem = etree.SubElement(smil_head, "metadata")
            etree.SubElement(metadata_elem, "meta", name="dc:title", content=f"Chapter {chapter_num+1}")
            etree.SubElement(metadata_elem, "meta", name="dtb:uid", content=str(uuid.uuid4()))
            etree.SubElement(metadata_elem, "meta", name="dtb:totalElapsedTime", content=f"{(chapter_start/1000):.3f}s")
            etree.SubElement(metadata_elem, "meta", name="dtb:duration", content=f"{chapter_duration_sec:.3f}s")

            # Add body with properly configured seq element
            smil_body = etree.SubElement(smil_root, "body")

            # Calculate the relative path from SMIL directory to the content file
            rel_content_path = os.path.relpath(
                os.path.join(package_dir, content_path_rel),
                smil_dir
            )

            seq = etree.SubElement(smil_body, "seq", {
                "id": f"seq-{chapter_num+1}",
                "{http://www.idpf.org/2007/ops}textref": rel_content_path
            })

            # Add par elements for each text fragment
            fragment_count = 0
            for i, fragment in enumerate(chapter_fragments):
                frag_start = fragment.get("start_time", 0)
                frag_end = fragment.get("end_time", 0)
                frag_id = f"frag-{chapter_num+1}-{i+1}"

                # Make sure we have a valid content reference
                if frag_id in fragment_to_content_map:
                    content_file, element_id = fragment_to_content_map[frag_id]

                    # Calculate relative paths
                    rel_audio_path = os.path.relpath(
                        os.path.join(audio_dir, audio_filename),
                        smil_dir
                    )

                    rel_content_path = os.path.relpath(
                        os.path.join(package_dir, content_file),
                        smil_dir
                    )

                    # Add par element
                    par = etree.SubElement(seq, "par", id=f"par-{frag_id}")

                    # Text element with valid fragment identifier
                    text = etree.SubElement(par, "text", id=f"text-{frag_id}",
                                         src=f"{rel_content_path}#{element_id}")

                    # Audio element
                    audio = etree.SubElement(par, "audio", id=f"audio-{frag_id}",
                                          src=f"{rel_audio_path}",
                                          clipBegin=f"{(frag_start/1000):.3f}s",
                                          clipEnd=f"{(frag_end/1000):.3f}s")

                    fragment_count += 1

            # Fix for RSC-005 error: If no fragments were added to the seq, add a dummy par element
            if fragment_count == 0:
                dummy_par = etree.SubElement(seq, "par", id=f"par-dummy-{chapter_num+1}")
                dummy_text = etree.SubElement(dummy_par, "text", id=f"text-dummy-{chapter_num+1}",
                                           src=f"{rel_content_path}")

                # Add a silent audio clip that's very short
                dummy_audio = etree.SubElement(dummy_par, "audio", id=f"audio-dummy-{chapter_num+1}",
                                            src=f"{os.path.relpath(os.path.join(audio_dir, audio_filename), smil_dir)}",
                                            clipBegin="0s",
                                            clipEnd="0.001s")

                print(f"Warning: Added dummy fragment to empty chapter {chapter_num+1}")

            # Write SMIL file
            smil_tree = etree.ElementTree(smil_root)
            with open(smil_path, 'wb') as f:
                f.write(etree.tostring(smil_root, pretty_print=True,
                                     xml_declaration=True, encoding="utf-8"))

            # Add SMIL file to manifest (or update if already exists)
            existing_smil = package_tree.xpath(f"//opf:item[@id='{smil_id}']", namespaces=ns)
            if existing_smil:
                existing_smil[0].set("href", f"smil/{smil_filename}")
                existing_smil[0].set("media-type", "application/smil+xml")
            else:
                smil_element = etree.SubElement(manifest, "{http://www.idpf.org/2007/opf}item")
                smil_element.set("id", smil_id)
                smil_element.set("href", f"smil/{smil_filename}")
                smil_element.set("media-type", "application/smil+xml")

            # Add duration metadata for this SMIL file
            smil_duration_meta = etree.SubElement(metadata, "{http://www.idpf.org/2007/opf}meta")
            smil_duration_meta.set("property", "media:duration")
            smil_duration_meta.set("refines", f"#{smil_id}")
            smil_duration_meta.text = f"{chapter_duration_sec:.3f}s"

            # Update content file to add media-overlay attributes
            if os.path.exists(os.path.join(package_dir, content_path_rel)):
                # Add media-overlay attribute to the item in the manifest
                manifest_item = package_tree.xpath(f"//opf:item[@id='{content_id}']", namespaces=ns)
                if manifest_item:
                    manifest_item[0].set("media-overlay", smil_id)

        # Verify that the sum of SMIL durations matches the total duration
        total_smil_duration = sum(smil_durations)
        if abs(total_smil_duration - total_duration_sec) > 0.1:  # Allow small rounding difference
            print(f"Warning: Sum of SMIL durations ({total_smil_duration:.3f}s) doesn't match total duration ({total_duration_sec:.3f}s)")
            # Update the total duration to match sum of SMIL durations
            total_duration_element = package_tree.xpath("//opf:meta[@property='media:duration' and not(@refines)]", namespaces=ns)
            if total_duration_element:
                total_duration_element[0].text = f"{total_smil_duration:.3f}s"

        # Save modified package file
        with open(package_path, 'wb') as f:
            f.write(etree.tostring(package_tree, pretty_print=True,
                                 xml_declaration=True, encoding="utf-8"))

        # Create CSS file for media overlay styling if it doesn't exist
        css_dir = os.path.join(package_dir, "css")
        os.makedirs(css_dir, exist_ok=True)
        css_path = os.path.join(css_dir, "media-overlay.css")

        if not os.path.exists(css_path):
            with open(css_path, 'w') as f:
                f.write("""
                .reading {
                    background-color: #ffff99 !important;
                    color: #000000 !important;
                }
                .mo-active {
                    background-color: #ffff99 !important;
                    color: #000000 !important;
                }
                """)

            # Add CSS file to manifest
            css_id = "media-overlay-css"
            existing_css = package_tree.xpath(f"//opf:item[@id='{css_id}']", namespaces=ns)
            if not existing_css:
                css_element = etree.SubElement(manifest, "{http://www.idpf.org/2007/opf}item")
                css_element.set("id", css_id)
                css_element.set("href", "css/media-overlay.css")
                css_element.set("media-type", "text/css")

                # Save updated package file again
                with open(package_path, 'wb') as f:
                    f.write(etree.tostring(package_tree, pretty_print=True,
                                         xml_declaration=True, encoding="utf-8"))

        # Create new EPUB file
        print(f"Creating enhanced EPUB with media overlay: {output_path}")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add mimetype file first, uncompressed
            mimetype_path = os.path.join(temp_dir, "mimetype")
            if not os.path.exists(mimetype_path):
                with open(mimetype_path, 'w') as f:
                    f.write("application/epub+zip")
                zipf.write(mimetype_path, "mimetype", zipfile.ZIP_STORED)
            else:
                zipf.write(mimetype_path, "mimetype", zipfile.ZIP_STORED)

            # Add all other files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == "mimetype":
                        continue  # Already added

                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, rel_path)

        # Clean up
        shutil.rmtree(temp_dir)

        print(f"Successfully created EPUB with media overlay: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error adding media overlay to EPUB: {e}")
        traceback.print_exc()
        return None
