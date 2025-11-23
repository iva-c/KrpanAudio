"""
Helper data models for classic_parsing.py.
Define small dataclasses for recognized text, images, and per-page
results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass
class TextElement:
    page_index: int  # 0-based page index in the PDF
    bbox: BBox  # (x0, y0, x1, y1)
    text: str
    block_num: int
    par_num: int
    line_num: int
    word_num: int


@dataclass
class ImageElement:
    page_index: int
    bbox: BBox  # (x0, y0, x1, y1)
    image_bytes: bytes  # PNG bytes


@dataclass
class PageResult:
    """
    OCR + optional full-page image for a single scanned page.
    """

    page_index: int  # 0-based
    image_bytes: bytes  # full page image (original rendered PNG)
    width: int  # rendered width in pixels
    height: int  # rendered height in pixels
    texts: List[TextElement]  # recognized words
    images: List[ImageElement]  # full-page image only if page has no text


@dataclass
class TextAndImages:
    """
    In-memory representation of the extracted document:
    - text: raw text with --- Page N --- markers and <image_n> placeholders
    - images: { "image_1": base64_string, ... }
    """

    text: str
    images: Dict[str, str]
