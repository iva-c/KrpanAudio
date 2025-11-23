"""
Helper functions for classic_parsing.py.
Processing pipeline: extract text and images from digital documents (PDF and Word).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import fitz
from .data_models import BBox, ImageElement, PageResult, TextElement
from docx import Document
from PIL import Image
from .scan_pipeline import render_page_to_rgb

# ========================== Digital PDF helpers ==========================


def _extract_text_elements_from_page(
    page: fitz.Page,
    page_index: int,
    width_px: int,
    height_px: int,
) -> List[TextElement]:
    """
    Use PyMuPDF 'words' output to create TextElement objects.

    words items look like:
        (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    Coordinates are in PDF points, so we scale to the rendered pixel space.
    """
    rect = page.rect
    sx = width_px / rect.width if rect.width > 0 else 1.0
    sy = height_px / rect.height if rect.height > 0 else 1.0

    words = page.get_text("words")
    texts: List[TextElement] = []

    for x0_pt, y0_pt, x1_pt, y1_pt, text, block_no, line_no, word_no in words:
        if not text or not text.strip():
            continue

        x0 = int(round(x0_pt * sx))
        y0 = int(round(y0_pt * sy))
        x1 = int(round(x1_pt * sx))
        y1 = int(round(y1_pt * sy))
        bbox: BBox = (x0, y0, x1, y1)

        # PyMuPDF words doesn't carry paragraph info -> par_num=0
        texts.append(
            TextElement(
                page_index=page_index,
                bbox=bbox,
                text=text.strip(),
                block_num=int(block_no),
                par_num=0,
                line_num=int(line_no),
                word_num=int(word_no),
            )
        )

    return texts


def _extract_image_elements_from_page(
    page: fitz.Page,
    page_index: int,
    width_px: int,
    height_px: int,
) -> List[ImageElement]:
    """
    Extract inline images (with bounding boxes) from a digital PDF page.

    Uses page.get_text("dict") image blocks:
      - "type" == 1 for image blocks
      - "bbox" in PDF coordinates
      - "image" raw bytes (jpg/png/etc.)
    """
    rect = page.rect
    sx = width_px / rect.width if rect.width > 0 else 1.0
    sy = height_px / rect.height if rect.height > 0 else 1.0

    d = page.get_text("dict")
    blocks = d.get("blocks", [])

    images: List[ImageElement] = []

    for b in blocks:
        if b.get("type") != 1:
            continue  # only image blocks

        x0_pt, y0_pt, x1_pt, y1_pt = b["bbox"]
        x0 = int(round(x0_pt * sx))
        y0 = int(round(y0_pt * sy))
        x1 = int(round(x1_pt * sx))
        y1 = int(round(y1_pt * sy))
        bbox: BBox = (x0, y0, x1, y1)

        raw_bytes: bytes = b["image"]

        # Normalize to PNG bytes
        try:
            pil = Image.open(io.BytesIO(raw_bytes))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        except Exception:
            img_bytes = raw_bytes

        images.append(
            ImageElement(
                page_index=page_index,
                bbox=bbox,
                image_bytes=img_bytes,
            )
        )

    return images


def process_pdf(pdf_path: Path) -> List[PageResult]:
    """
    Process a *digital* PDF:
      - Render each page (same geometry as scan pipeline).
      - Extract words (+ bounding boxes) as TextElement.
      - Extract inline images (+ bounding boxes) as ImageElement,
        even if the page also has text.
      - Return List[PageResult].
    """
    doc = fitz.open(pdf_path)
    pages: List[PageResult] = []

    try:
        for idx, page in enumerate(doc):
            rgb, png_bytes = render_page_to_rgb(page)
            h, w = rgb.shape[:2]

            texts = _extract_text_elements_from_page(
                page=page,
                page_index=idx,
                width_px=w,
                height_px=h,
            )
            images = _extract_image_elements_from_page(
                page=page,
                page_index=idx,
                width_px=w,
                height_px=h,
            )

            pages.append(
                PageResult(
                    page_index=idx,
                    image_bytes=png_bytes,
                    width=w,
                    height=h,
                    texts=texts,
                    images=images,
                )
            )
    finally:
        doc.close()

    return pages


# ============================= Word pipeline =============================


def _make_blank_page_png(width: int, height: int) -> bytes:
    """
    Create a simple white PNG used as the 'page image' for Word documents.
    This keeps PageResult.image_bytes always defined.
    """
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_docx(docx_path: Path) -> List[PageResult]:
    """
    Basic Word (.docx) processing:

    - Treat the whole document as a single logical 'page'.
    - Read paragraphs using python-docx.
    - Create TextElement objects with synthetic bounding boxes so they can
      be assembled into lines by io_utils.
    - Extract inline images (doc.inline_shapes) and create ImageElement
      objects with synthetic bounding boxes.

    Limitations:
    - We don't know true page layout or exact image positions; bboxes are
      synthetic but consistent enough for later ordering.
    """
    doc = Document(docx_path)

    # Synthetic A4-like page at ~200 dpi:
    width_px = 1654
    height_px = 2338
    page_index = 0

    image_bytes = _make_blank_page_png(width_px, height_px)
    texts: List[TextElement] = []

    y_step = 24
    x_step = 12
    margin_x = 50
    margin_y = 50

    current_line = 0

    # ---- TEXT: paragraphs -> TextElement ----
    for par_index, par in enumerate(doc.paragraphs):
        if not par.text.strip():
            current_line += 1
            continue

        words = par.text.split()
        if not words:
            current_line += 1
            continue

        y0 = margin_y + current_line * y_step

        for word_index, word in enumerate(words, start=1):
            x0 = margin_x + (word_index - 1) * x_step
            w_px = max(10, len(word) * 8)
            h_px = 18
            x1 = x0 + w_px
            y1 = y0 + h_px
            bbox: BBox = (x0, y0, x1, y1)

            texts.append(
                TextElement(
                    page_index=page_index,
                    bbox=bbox,
                    text=word,
                    block_num=0,
                    par_num=par_index,
                    line_num=current_line,
                    word_num=word_index,
                )
            )

        current_line += 1

    # ---- IMAGES: inline shapes -> ImageElement ----
    images: List[ImageElement] = []

    # Place images *after* the last text line, each on its own synthetic line.
    base_line_for_images = current_line + 1
    for img_idx, inline_shape in enumerate(doc.inline_shapes, start=0):
        # Grab image bytes via relationship id
        try:
            rId = inline_shape._inline.graphic.graphicData.pic.blipFill.blip.embed
            image_part = doc.part.related_parts[rId]
            img_bytes = image_part.blob
        except Exception:
            # If something goes wrong, skip this image
            continue

        # Synthetic bbox somewhere below the text block
        y0 = margin_y + (base_line_for_images + img_idx) * y_step
        h_px = 120
        x0 = margin_x
        x1 = x0 + 160
        y1 = y0 + h_px
        bbox: BBox = (x0, y0, x1, y1)

        # Optionally normalize to PNG
        try:
            pil = Image.open(io.BytesIO(img_bytes))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            png_bytes = buf.getvalue()
        except Exception:
            png_bytes = img_bytes

        images.append(
            ImageElement(
                page_index=page_index,
                bbox=bbox,
                image_bytes=png_bytes,
            )
        )

    page = PageResult(
        page_index=page_index,
        image_bytes=image_bytes,
        width=width_px,
        height=height_px,
        texts=texts,
        images=images,
    )

    return [page]


# ============================ Unified entrypoint =========================


def process_digital(path: Path) -> List[PageResult]:
    """
    Unified entrypoint for digital documents:

        - If .pdf  -> process_pdf(...)
        - If .docx / .doc -> process_docx(...)

    Returns:
        List[PageResult] compatible with scan_pipeline,
        so io_utils.save_images_and_text works for both.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return process_pdf(path)

    if suffix in {".docx", ".doc"}:
        return process_docx(path)

    raise ValueError(f"Unsupported file type for digital_pipeline: {suffix!r}")
