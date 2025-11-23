"""
Helper functions for classic_parsing.py.
Assemble textual output and in-memory images.
"""

from __future__ import annotations

import base64
from typing import Dict, List

from data_models import PageResult, TextAndImages, TextElement


def assemble_text_lines(texts: List[TextElement]) -> List[str]:
    """
    Assemble TextElements into text lines based on (block_num, par_num, line_num),
    preserving spatial order.

    This mirrors the logic used in scan_pipeline, but is kept here so
    io_utils can be used by any pipeline without circular imports.
    """
    if not texts:
        return []

    # Group by (block, par, line)
    groups: dict[tuple[int, int, int], list[TextElement]] = {}
    for t in texts:
        key = (t.block_num, t.par_num, t.line_num)
        groups.setdefault(key, []).append(t)

    # Sort lines by vertical position (top y of first word)
    def line_y(key: tuple[int, int, int]) -> int:
        return min(te.bbox[1] for te in groups[key])

    sorted_keys = sorted(groups.keys(), key=line_y)

    lines: List[str] = []
    for key in sorted_keys:
        words = sorted(groups[key], key=lambda te: te.bbox[0])  # sort by x0
        line_text = " ".join(w.text for w in words if w.text.strip())
        if line_text.strip():
            lines.append(line_text.strip())

    return lines


def build_text_and_images(pages: List[PageResult]) -> TextAndImages:
    """
    Build a raw marked text and an in-memory image dictionary from PageResult list.

    Text format example:
        --- Page 1 ---
        Some text line
        Another line
        <image_1>

        --- Page 2 ---
        <image_2>

    Images dict:
        {
            "image_1": "<base64-encoded PNG>",
            "image_2": "<base64-encoded PNG>",
            ...
        }

    This replaces the old save_images_and_text() that wrote files.
    """
    lines: List[str] = []
    images: Dict[str, str] = {}
    image_counter = 1

    for page in pages:
        # Page header (1-based for humans)
        lines.append(f"--- Page {page.page_index + 1} ---")

        # Text from OCR / digital extraction
        page_lines = assemble_text_lines(page.texts)
        lines.extend(page_lines)

        # Full-page (or region) images
        for img_elem in page.images:
            image_name = f"image_{image_counter}"

            # Base64 encode the bytes (ASCII-safe string)
            b64 = base64.b64encode(img_elem.image_bytes).decode("ascii")
            images[image_name] = b64

            # Insert placeholder into the text
            lines.append(f"<{image_name}>")

            image_counter += 1

        # Separate pages visually
        lines.append("")

    raw_text = "\n".join(lines).rstrip() + "\n"
    return TextAndImages(text=raw_text, images=images)
