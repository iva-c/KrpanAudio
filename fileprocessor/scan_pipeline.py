"""
Helper functions for classic_parsing.py.
Scanning pipeline: run OCR to generate images and text from scanned pages.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from .data_models import BBox, ImageElement, PageResult, TextElement
from PIL import Image
from pytesseract import Output

# ======================================================================
# Rendering
# ======================================================================


def render_page_to_rgb(
    page: fitz.Page,
    target_dpi: int = 200,
    max_side: int = 2200,
) -> Tuple[np.ndarray, bytes]:
    """
    Render a PyMuPDF page to an RGB NumPy array, capping the largest side.

    Returns:
        (rgb, png_bytes)
        - rgb: HxWx3 uint8 array in RGB
        - png_bytes: encoded PNG bytes of the page rendering
    """
    base_zoom = target_dpi / 72.0
    rect = page.rect
    w0 = rect.width * base_zoom
    h0 = rect.height * base_zoom
    max_dim = max(w0, h0)

    scale = 1.0
    if max_dim > max_side:
        scale = max_side / max_dim

    zoom = base_zoom * scale
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)  # pix.n should be 3
    rgb = img.copy()  # already RGB

    return rgb, png_bytes


# ======================================================================
# Background normalization & despeckling
# ======================================================================


def normalize_background_to_white(
    img_cv: np.ndarray,
    bg_border_frac: float = 0.12,
    color_thresh: float = 16.0,
) -> np.ndarray:
    """
    Estimate background color from page borders and push background-like pixels to white.
    Returns a new BGR image (OpenCV channel order).
    """
    h, w, _ = img_cv.shape
    img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)

    border = int(min(h, w) * bg_border_frac)
    border = max(border, 1)

    top_strip = img_lab[0:border, :, :]
    bottom_strip = img_lab[h - border : h, :, :]
    left_strip = img_lab[:, 0:border, :]
    right_strip = img_lab[:, w - border : w, :]

    border_pixels = np.concatenate(
        [
            top_strip.reshape(-1, 3),
            bottom_strip.reshape(-1, 3),
            left_strip.reshape(-1, 3),
            right_strip.reshape(-1, 3),
        ],
        axis=0,
    )

    bg_L, bg_a, bg_b = np.median(border_pixels, axis=0)
    bg_color = np.array([bg_L, bg_a, bg_b], dtype=np.float32)

    lab_flat = img_lab.reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(lab_flat - bg_color[None, :], axis=1)

    mask_bg_flat = dist < color_thresh
    # OpenCV LAB white ≈ [255, 128, 128]
    lab_flat[mask_bg_flat] = np.array([255, 128, 128], dtype=np.float32)

    img_lab_bal = lab_flat.reshape(h, w, 3).astype(np.uint8)
    img_balanced = cv2.cvtColor(img_lab_bal, cv2.COLOR_LAB2BGR)
    return img_balanced


def despeckle_binary(
    bin_img: np.ndarray,
    max_speckle_area: int = 120,
) -> np.ndarray:
    """
    Remove tiny connected components from a binary image (0/255).
    """
    bin_img = (bin_img > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )

    cleaned = np.zeros_like(bin_img)
    for label in range(1, num_labels):  # 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= max_speckle_area:
            cleaned[labels == label] = 255
    return cleaned


# ======================================================================
# Small OCR helpers
# ======================================================================


def _clahe_unsharp(gray: np.ndarray) -> np.ndarray:
    """
    Light CLAHE + gentle unsharp (for OCR input only; never for gating).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(g, 1.3, blur, -0.3, 0)
    return sharp


def _entropy(gray: np.ndarray) -> float:
    """
    Shannon entropy (0..~8) of a grayscale image.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _edge_density(gray: np.ndarray) -> float:
    """
    Fraction of edge pixels (Canny) on lightly blurred grayscale.
    """
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 60, 180)
    return float((edges > 0).mean())


# ======================================================================
# Heuristic thresholds (unchanged behavior)
# ======================================================================

VERY_DARK_THRESHOLD = 0.50  # >50% dark pixels => heavy graphics
MIN_DARK_FOR_CONTENT = 0.11  # if no text but >11% dark, likely content

BLANK_ENTROPY_MAX = 3.6
BLANK_EDGE_MAX = 0.025
BLANK_DARK_MAX = 0.22

# Light-photo detector (bright page with texture & broad midtones)
LIGHT_PHOTO_MEAN_MIN = 165.0
LIGHT_PHOTO_MEAN_MAX = 245.0
LIGHT_PHOTO_STD_MIN = 8.0
LIGHT_PHOTO_EDGE_MIN = 0.015
LIGHT_PHOTO_EDGE_MAX = 0.12
LIGHT_PHOTO_MIDTONE_MIN = 0.35
LIGHT_PHOTO_DARK_MIN = 0.006
LIGHT_PHOTO_DARK_MAX = 0.18

# OCR text "strength" guard: if OCR finds only a few tiny bits, treat as weak text
WEAK_TEXT_MIN_WORDS = 8
WEAK_TEXT_MIN_AREA_FRAC = 0.004  # 0.4% of page area


# ======================================================================
# OCR + Page logic
# ======================================================================


def _ocr_single_page(
    thresh: np.ndarray,
    page_index: int,
    lang: str,
    config: str,
    conf_threshold: float = 40.0,
) -> List[TextElement]:
    """
    Run Tesseract image_to_data on a thresholded page image and return TextElement objects.
    """
    data = pytesseract.image_to_data(
        thresh,
        output_type=Output.DICT,
        lang=lang,
        config=config,
    )

    texts: List[TextElement] = []
    n = len(data["text"])

    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if conf < conf_threshold:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w_box = int(data["width"][i])
        h_box = int(data["height"][i])
        if w_box <= 0 or h_box <= 0:
            continue

        block_num = int(data["block_num"][i])
        par_num = int(data["par_num"][i])
        line_num = int(data["line_num"][i])
        word_num = int(data["word_num"][i])

        bbox: BBox = (x, y, x + w_box, y + h_box)

        texts.append(
            TextElement(
                page_index=page_index,
                bbox=bbox,
                text=text,
                block_num=block_num,
                par_num=par_num,
                line_num=line_num,
                word_num=word_num,
            )
        )

    texts.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
    return texts


def _make_full_page_image_element(
    page_index: int,
    width: int,
    height: int,
    image_bytes: bytes,
) -> ImageElement:
    """
    Create a single full-page ImageElement (bbox = whole page).
    """
    bbox: BBox = (0, 0, width, height)
    return ImageElement(
        page_index=page_index,
        bbox=bbox,
        image_bytes=image_bytes,
    )


def process_page(
    page: fitz.Page,
    page_index: int,
    dpi: int = 200,
    lang: str = "slv+eng",
    psm: int = 4,
    max_side_px: int = 2200,
) -> PageResult:
    """
    OCR a single scanned page:

    Pipeline overview:
      - Render and slightly crop borders.
      - Normalize background (gentle threshold) to preserve light photos.
      - Compute all gating stats on the *balanced but unenhanced* grayscale.
      - EARLY GATES:
          * Very dark page -> image (unless blank-textured).
          * Light-photo page -> image (unless blank-textured).
      - OCR (apply CLAHE/unsharp only for OCR path).
      - If OCR finds no text OR only weak text, and page still looks like content,
        emit full-page image (unless blank-textured).
    """
    # 1) Render
    rgb_full, _ = render_page_to_rgb(page, target_dpi=dpi, max_side=max_side_px)

    # Border crop
    pil_img = Image.fromarray(rgb_full).convert("RGB")
    if pil_img.width > 10 and pil_img.height > 10:
        pil_img = pil_img.crop((5, 5, pil_img.width - 5, pil_img.height - 5))

    width, height = pil_img.size
    page_area = float(width * height)

    img_np_rgb = np.array(pil_img)  # RGB
    img_cv = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)  # BGR

    # 2) Normalize background (gentle)
    img_cv_bal = normalize_background_to_white(
        img_cv,
        bg_border_frac=0.12,
        color_thresh=16.0,
    )
    img_np_bal_rgb = cv2.cvtColor(img_cv_bal, cv2.COLOR_BGR2RGB)

    # 3) Base grayscale for all heuristics (no CLAHE/unsharp)
    gray_base = cv2.cvtColor(img_np_bal_rgb, cv2.COLOR_RGB2GRAY)
    blur_base = cv2.medianBlur(gray_base, 5)
    _, thresh_base = cv2.threshold(
        blur_base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh_base = despeckle_binary(thresh_base, max_speckle_area=60)

    dark_ratio_base = float(np.mean(thresh_base == 0))
    mean_i = float(gray_base.mean())
    std_i = float(gray_base.std())
    ent = _entropy(gray_base)
    edg = _edge_density(gray_base)
    midtone_ratio = float(((gray_base >= 140) & (gray_base <= 220)).mean())

    looks_blank_textured = (
        (ent <= BLANK_ENTROPY_MAX)
        and (edg <= BLANK_EDGE_MAX)
        and (dark_ratio_base <= BLANK_DARK_MAX)
    )

    # --- Early heavy-graphics gate ---
    if (dark_ratio_base > VERY_DARK_THRESHOLD) and (not looks_blank_textured):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        page_bytes = buf.getvalue()
        img_elem = ImageElement(
            page_index=page_index, bbox=(0, 0, width, height), image_bytes=page_bytes
        )
        return PageResult(
            page_index=page_index,
            image_bytes=page_bytes,
            width=width,
            height=height,
            texts=[],
            images=[img_elem],
        )

    # --- Early light-photo gate (before OCR) ---
    looks_light_photo = (
        (LIGHT_PHOTO_MEAN_MIN <= mean_i <= LIGHT_PHOTO_MEAN_MAX)
        and (std_i >= LIGHT_PHOTO_STD_MIN)
        and (LIGHT_PHOTO_EDGE_MIN <= edg <= LIGHT_PHOTO_EDGE_MAX)
        and (midtone_ratio >= LIGHT_PHOTO_MIDTONE_MIN)
        and (LIGHT_PHOTO_DARK_MIN <= dark_ratio_base <= LIGHT_PHOTO_DARK_MAX)
        and (not looks_blank_textured)
    )
    if looks_light_photo:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        page_bytes = buf.getvalue()
        img_elem = _make_full_page_image_element(page_index, width, height, page_bytes)
        return PageResult(
            page_index=page_index,
            image_bytes=page_bytes,
            width=width,
            height=height,
            texts=[],
            images=[img_elem],
        )

    # 4) OCR (enhancements only for OCR path)
    gray_ocr = _clahe_unsharp(gray_base)
    blur_ocr = cv2.medianBlur(gray_ocr, 5)
    _, thresh_ocr = cv2.threshold(blur_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_ocr = despeckle_binary(thresh_ocr, max_speckle_area=120)

    config = f"--oem 1 --psm {psm}"
    texts = _ocr_single_page(
        thresh=thresh_ocr,
        page_index=page_index,
        lang=lang,
        config=config,
    )
    has_text = len(texts) > 0

    # --- Weak text guard ---
    text_area = 0.0
    for t in texts:
        x0, y0, x1, y1 = t.bbox
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        text_area += float(w * h)
    text_area_frac = text_area / (page_area + 1e-9)

    weak_text = (len(texts) < WEAK_TEXT_MIN_WORDS) or (
        text_area_frac < WEAK_TEXT_MIN_AREA_FRAC
    )

    # 5) If no/weak text but page still looks like content → image
    if (not has_text or weak_text) and (not looks_blank_textured):
        if (dark_ratio_base > MIN_DARK_FOR_CONTENT) or (
            (LIGHT_PHOTO_MEAN_MIN <= mean_i <= LIGHT_PHOTO_MEAN_MAX)
            and (std_i >= LIGHT_PHOTO_STD_MIN)
            and (LIGHT_PHOTO_EDGE_MIN <= edg <= LIGHT_PHOTO_EDGE_MAX)
            and (midtone_ratio >= LIGHT_PHOTO_MIDTONE_MIN)
            and (LIGHT_PHOTO_DARK_MIN <= dark_ratio_base <= LIGHT_PHOTO_DARK_MAX)
        ):
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            page_bytes = buf.getvalue()
            img_elem = _make_full_page_image_element(
                page_index, width, height, page_bytes
            )
            return PageResult(
                page_index=page_index,
                image_bytes=page_bytes,
                width=width,
                height=height,
                texts=[],
                images=[img_elem],
            )

    # Otherwise: normal text page (or truly blank-textured)
    buf_full = io.BytesIO()
    pil_img.save(buf_full, format="PNG")
    full_page_bytes = buf_full.getvalue()

    return PageResult(
        page_index=page_index,
        image_bytes=full_page_bytes,
        width=width,
        height=height,
        texts=texts,
        images=[],
    )


def process_pdf(
    pdf_path: Path,
    dpi: int = 200,
    lang: str = "slv+eng",
    psm: int = 4,
) -> List[PageResult]:
    """
    Process the entire scanned PDF and return a list of PageResult objects.
    """
    doc = fitz.open(pdf_path)
    pages: List[PageResult] = []
    try:
        for idx, page in enumerate(doc):
            pages.append(
                process_page(
                    page,
                    page_index=idx,
                    dpi=dpi,
                    lang=lang,
                    psm=psm,
                )
            )
    finally:
        doc.close()
    return pages
