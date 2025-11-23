"""Routing helpers for document processing.

This module exposes small, well-documented functions used by the
application to detect input file type and to dispatch files to the
appropriate processing pipeline. It is intended to be imported by
other code (web backends, CLI, tests) and therefore has no side effects
on import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal

import fitz  # PyMuPDF

from cleaning import clean_marked_text
from digital_pipeline import process_digital

# NEW: imports for the in-memory result
from io_utils import TextAndImages, build_text_and_images
from data_models import PageResult
from scan_pipeline import process_pdf as process_scanned_pdf

FileKind = Literal["scan_pdf", "digital_pdf", "word"]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def is_classical(path: str | Path) -> bool:
    """
    Return True if the input looks like a 'classical' document we handle at the top level:
    a PDF (.pdf) or a Word document (.docx). Case-insensitive; checks extension only.
    """
    suffix = Path(path).suffix.lower()
    return suffix in {".pdf", ".docx"}


def is_digital_pdf(pdf_path: Path, text_threshold: int = 5) -> bool:
    """
    Decide whether a PDF is 'digital' (has embedded text) or a scanned image.

    Heuristic:
        - open PDF
        - inspect ONLY the first page
        - extract text with page.get_text("text")
        - if len(text) > text_threshold => digital PDF

    If anything is weird/empty, we fall back to treating it as scanned.
    """
    doc = fitz.open(pdf_path)

    try:
        if len(doc) == 0:
            # Empty / corrupted: safest to treat as scan
            return False

        first_page = doc[0]
        text = first_page.get_text("text") or ""
        return len(text) > text_threshold
    finally:
        doc.close()


def detect_file_kind(path: Path) -> FileKind:
    """
    Detect whether the input is:
        - 'scan_pdf'
        - 'digital_pdf'
        - 'word'
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if is_digital_pdf(path):
            return "digital_pdf"
        else:
            return "scan_pdf"

    if suffix in {".docx", ".doc"}:
        return "word"

    raise ValueError(
        f"Unsupported file type: {suffix!r} (only .pdf, .docx, .doc supported)"
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_document(path: str | Path) -> List[PageResult]:
    """
    High-level entry point:

        - Detect file kind (scan_pdf / digital_pdf / word)
        - Call the appropriate pipeline
        - Return List[PageResult] with a unified structure
    """
    path = Path(path)
    kind = detect_file_kind(path)

    if kind == "scan_pdf":
        return process_scanned_pdf(path)
    elif kind in {"digital_pdf", "word"}:
        return process_digital(path)

    # Should never get here because detect_file_kind would raise earlier
    raise RuntimeError(f"Unhandled file kind: {kind!r}")


# ---------------------------------------------------------------------------
# New high-level API: cleaned TextAndImages
# ---------------------------------------------------------------------------


def route_and_clean_document(
    path: str | Path, detect_headers: bool = True
) -> TextAndImages:
    """
    End-to-end entry point for the rest of the app:

        - Route the document to the correct pipeline.
        - Build a raw text + images representation in-memory.
        - Clean the text (garbage removal + image relocation).
        - Return TextAndImages with CLEANED text and base64-encoded images.

    The returned object is ready for TTS / frontend:
        result.text   -> readable text with <image_n> placeholders
        result.images -> {"image_1": base64_string, ...}
    """
    pages: List[PageResult] = route_document(path)

    # Build raw marked text (--- Page N --- + <image_n>) and images dict
    assembled = build_text_and_images(pages)

    # Clean text using the cleaning pipeline
    cleaned_text = clean_marked_text(assembled.text, detect_headers=detect_headers)

    # Return the same structure, but with cleaned text
    return TextAndImages(text=cleaned_text, images=assembled.images)


def classic_doc_parsing(
    path: str | Path, detect_headers: bool = True
) -> tuple[str, Dict[str, str]]:
    """
    Adapter for the rest of the app, so it can do:
        text, images = classic_doc_parsing(path)
    """
    result = route_and_clean_document(path, detect_headers=detect_headers)
    return result.text, result.images
