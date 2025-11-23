# file: cleaning.py
"""
OCR/PDF → TTS cleaner with robust image relocation.

Key ideas
---------
- <image_n> placeholders are NEVER removed.
- Images move FORWARD to the NEXT sentence end via buffering,
  except when they are in a "safe" block position:
    * before any real text in the document (front-matter image block), or
    * immediately after a completed sentence.
- Sentence ends = '.', '!', '?', '…' (possibly repeated) with trailing spaces/closers.
- Punctuation junk is cleaned without destroying short words near quotes.
- Lines inside a page are flattened into a single paragraph (newlines → spaces).
- Repeated headers/footers, page numbers, OCR garbage pages are dropped,
  but pages containing images are always preserved.
- Optional: drop standalone single-letter lines (default: only 'a').
- Final pass enforces BLANK LINES around every <image_n> and verifies that
  all image tags are preserved in count and order.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Set

# ============================================================================
# 0. Configuration
# ============================================================================

# Single-letter lines to drop (OCR orphans). Normalized & case-folded.
DROP_SINGLE_LETTER_LINES: set[str] = {"a"}


# ============================================================================
# 1. Page-level statistics & classification
# ============================================================================

def page_stats(text: str) -> Dict[str, float]:
    """
    Compute simple statistics about a page for classification heuristics.
    """
    total_chars = len(text)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    whitespace = sum(c.isspace() for c in text)
    weird = sum(c in "#@$%&*=+~^_|<>�" for c in text)
    words = text.split()

    def ratio(x: int) -> float:
        return x / total_chars if total_chars else 0.0

    return {
        "num_chars": float(total_chars),
        "num_words": float(len(words)),
        "ratio_letters": ratio(letters),
        "ratio_digits": ratio(digits),
        "ratio_whitespace": ratio(whitespace),
        "ratio_weird": ratio(weird),
    }


# ----------------------------- Metadata / promo detection --------------------

METADATA_KEYWORDS = [
    "isbn",
    "copyright",
    "avtorska pravica",
    "all rights reserved",
    "vse pravice pridržane",
    "printed in",
    "natisnjeno v",
    "publisher",
    "izdajatelj",
    "published by",
    "izdano pri",
    "first published",
    "prvič objavljeno",
    "prva izdaja",
    "cover design",
    "oblikovanje naslovnice",
    "oblikovanje platnice",
    "typeset",
    "pripravljeno za tisk",
    "library of congress",
    "kongresna knjižnica",
    "cataloguing",
    "katalogizacija",
    "edition",
    "izdaja",
    "no part of this",
    "noben del tega",
]

PROMO_KEYWORDS = [
    "also available",
    "available on",
    "available from",
    "available at",
    "also by",
    "other books",
    "you may also like",
    "find more",
    "order online",
    "buy",
    "na voljo",
    "na voljo tudi",
    "na voljo pri",
    "na voljo na",
    "na voljo v",
    "tudi od avtorja",
    "druge knjige",
    "druga dela",
    "morda vam bo všeč",
    "poiščite več",
    "naročite na spletu",
    "kupi",
    "kupi zdaj",
]


def is_metadata_page(
    text: str,
    stats: Dict[str, float],
    page_index: int,
    total_pages: int,
    edge_pages: int = 7,
    max_words: int = 200,
) -> bool:
    """
    Detect front/back matter (ISBN page, rights, publisher info, promo lists).

    Heuristics:
    - Never treat pages containing <image_n> as metadata (we must keep image pages).
    - Only classify as metadata on early/late pages (within `edge_pages` of ends).
    - Require at least one strong signal: URL/email/promo phrase/metadata keyword.
    - Do NOT rely solely on “many digits” to avoid killing normal pages.
    """
    if "<image_" in text:
        return False

    # Only check near edges of the document
    if not (page_index < edge_pages or page_index >= total_pages - edge_pages):
        return False

    t = text.lower()
    num_words = int(stats["num_words"])

    has_url = "www." in t or "http://" in t or "https://" in t
    has_email = "@" in t
    has_promo = any(p in t for p in PROMO_KEYWORDS)
    has_meta_kw = any(kw in t for kw in METADATA_KEYWORDS)

    # Too long → unlikely to be just metadata
    if num_words > max_words:
        return False

    if has_url or has_email or has_promo or has_meta_kw:
        return True

    return False


def is_image_like_page(
    stats: Dict[str, float],
    max_words: int = 10,
    max_letters_ratio: float = 0.4,
) -> bool:
    """
    Very short, low-text pages (e.g. mostly image/graphics).
    Currently not used as a hard filter but useful for analysis.
    """
    return bool(
        stats["num_words"] <= max_words and stats["ratio_letters"] <= max_letters_ratio
    )


# ============================================================================
# 2. Repeated header / footer detection
# ============================================================================

def detect_repeated_lines(
    pages: Iterable[str],
    min_page_fraction: float = 0.4,
    max_length: int = 80,
) -> Set[str]:
    """
    Find lines that appear on many pages (e.g. running headers/footers).

    Only short lines (<= max_length) are considered, and we de-duplicate per page
    before counting so that a header repeated twice on the same page isn't
    overcounted.
    """
    pages = list(pages)
    if not pages:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        seen = set()
        for raw in page.splitlines():
            line = raw.strip()
            if not line or len(line) > max_length:
                continue
            if line not in seen:
                seen.add(line)
                counts[line] += 1

    threshold = max(1, int(min_page_fraction * len(pages)))
    return {line for line, c in counts.items() if c >= threshold}


# ============================================================================
# 3. Line-level helpers & garbage heuristics
# ============================================================================

_PAGE_NUMBER_RE = re.compile(r"^\d+\s*$")
_PAGE_X_RE = re.compile(r"^(?:[Pp]age|[Ss]tran)\s*\d+\s*$")
_ALL_CAPS_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 ,.'\-:;!?]{4,}$")
_IMAGE_TAG_RE = re.compile(r"<image_(\d+)>")
_FINAL_PUNCT = ".!?"
_CLOSERS = "»”'\"')]}"

# Kept mostly for clarity; sentence-end logic is in the relocator
_SENT_END_RE = re.compile(r"[.!?…]+(?:\s*[" + re.escape(_CLOSERS) + r"]*)")


def _is_all_caps_header(
    line: str,
    min_letters: int = 4,
    min_upper_ratio: float = 0.8,
) -> bool:
    """
    Detect SHOUTY headers: mostly uppercase, enough letters.
    """
    s = line.strip()
    letters = [c for c in s if c.isalpha()]
    if len(letters) < min_letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) >= min_upper_ratio


def _is_noisy_line(
    line: str,
    min_length: int = 15,
    min_letter_ratio: float = 0.3,
) -> bool:
    """
    Detect lines that are mostly non-letters (noise / artifacts).
    """
    s = line.strip()
    if len(s) < min_length:
        return False
    letters = sum(c.isalpha() for c in s)
    return (letters / len(s)) < min_letter_ratio if s else False


def _is_garbage_line(line: str) -> bool:
    """
    Detect lines that are mostly punctuation with very few letters.
    Image tags are never considered garbage.
    """
    s = line.strip()
    if not s:
        return False
    if _IMAGE_TAG_RE.fullmatch(s):
        return False

    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 5:
        return False
    if re.findall(r"[A-Za-zÀ-ž]{2,}", s):
        return False

    nonspace = [c for c in s if not c.isspace()]
    if not nonspace:
        return False

    punct = sum(c in ".,;:!?-–—'\"«»()[]{}|" for c in s)
    return (punct / len(nonspace)) >= 0.4


def _meaningful_word_fraction(text: str) -> float:
    """
    Ratio of “word-like” tokens to all tokens.
    """
    words = re.findall(r"[A-Za-zÀ-ž]{2,}", text)
    tokens = text.split()
    return len(words) / max(1, len(tokens))


def is_garbage_page(
    text: str,
    stats: Dict[str, float],
    max_weird_ratio: float = 0.25,
    max_letters_ratio: float = 0.30,
    min_chars: int = 20,
) -> bool:
    """
    Generic garbage detection: noise-dominated pages, random punctuation, etc.
    Image pages are handled separately and are not dropped by this function.
    """
    if stats["num_chars"] < min_chars and stats["ratio_letters"] < 0.2:
        return True
    if stats["ratio_weird"] > max_weird_ratio and stats["ratio_letters"] < max_letters_ratio:
        return True
    if stats["num_words"] <= 30 and _meaningful_word_fraction(text) < 0.5:
        return True

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        gar = sum(1 for ln in lines if _is_garbage_line(ln))
        if (gar / max(1, len(lines))) >= 0.6:
            return True

    return False


def _ensure_terminal_punctuation(text: str) -> str:
    """
    Ensure text ends with terminal punctuation IF:
    - there is actual word-like content (letters/digits),
    - and the text does not already end in . ! ? or ….

    This avoids creating stray '.' when we only have images/whitespace.
    """
    s = text.rstrip()
    if not s:
        return s

    # No word-like characters -> don't invent punctuation
    if not re.search(r"[A-Za-z0-9À-ž]", s):
        return s

    i = len(s) - 1
    while i >= 0 and (s[i].isspace() or s[i] in _CLOSERS):
        i -= 1
    if i >= 0 and (s[i] in _FINAL_PUNCT or s[i] == "…"):
        return s
    return s + "."


def _clean_punctuation_artifacts(text: str) -> str:
    """
    Clean punctuation/spacing without touching <image_n>.

    Operations:
    - Normalize «», <<, >> to simple quotes.
    - Fix dot duplication and weird '.,', '!.', etc.
    - Remove dash clutter around commas/semicolons/colons.
    - Remove standalone '|' artifacts.
    - Turn 'word : word' into 'word word'.
    - Strip pure punctuation "words" at boundaries.
    - Normalize spaces and newlines (without collapsing all newlines).
    """
    parts = re.split(r"(<image_\d+>)", text)
    for i in range(0, len(parts), 2):
        seg = parts[i]

        # Normalize guillemets / angle-quote junk
        seg = seg.replace("«", '"').replace("»", '"')
        seg = re.sub(r"<<+", '"', seg)
        seg = re.sub(r">>+", '"', seg)

        # Fix excessive dots etc.
        seg = re.sub(r"([,;:!?…])\.", r"\1", seg)
        seg = re.sub(r"([!?]+)\.", r"\1", seg)
        seg = re.sub(r"\.\s+\.", ".", seg)
        seg = re.sub(r"([!?…])\s+\.", r"\1", seg)

        # Remove dash clutter next to ,;:
        seg = re.sub(r"([,;:])\s*[-–—]+", r"\1", seg)
        seg = re.sub(r"[-–—]+\s*([,;:])", r"\1", seg)

        # Kill lone pipes
        seg = re.sub(r"\s*\|+\s*", " ", seg)

        # word : word -> word word
        seg = re.sub(r"(?<=\w)\s*:\s+(?=[A-Za-zÀ-ž])", " ", seg)

        # Strip pure punctuation "words" at boundaries
        seg = re.sub(r'(?:(?<=\s)|^)[,"\'«»(){}\[\]—–\-;:|]+(?=\s|$)', " ", seg)

        # Normalize spaces/newlines
        seg = re.sub(r"[ \t\u00A0]{2,}", " ", seg)
        seg = re.sub(r"\s+\n", "\n", seg)
        seg = re.sub(r"\n{3,}", "\n\n", seg)
        seg = re.sub(r"(^|\n)[ \t]+", r"\1", seg)

        parts[i] = seg

    return "".join(parts).strip()


def _drop_specific_single_letter_lines(text: str, letters: set[str]) -> str:
    """
    Drop lines that consist solely of specific single letters (e.g. lone 'a'),
    which are usually OCR artifacts. Image tags and non-alpha lines are kept.
    """
    if not letters:
        return text

    norm_letters = {unicodedata.normalize("NFC", ch).lower() for ch in letters}
    out: List[str] = []

    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if (
            len(s) == 1
            and s.isalpha()
            and unicodedata.normalize("NFC", s).lower() in norm_letters
        ):
            continue  # drop OCR orphan
        out.append(ln)

    return "\n".join(out)


def _force_blank_lines_around_images(text: str) -> str:
    """
    Ensure every <image_n> is surrounded by exactly one blank line above and below.
    """
    text = re.sub(r"\s*(<image_\d+>)\s*", r"\n\n\1\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(^|\n)[ \t]+", r"\1", text)
    return text.strip()


# ============================================================================
# 4. Global image relocation
# ============================================================================

def _relocate_images_forward_globally(text: str) -> str:
    """
    Globally relocate <image_n> tags to avoid splitting sentences, with rules:

    - While no real text (letters/digits) has been seen:
        → all images are treated as a front image block and emitted immediately.
    - After real text appears:
        - If an image is at a "block start" (i.e. immediately after a completed
          sentence '.', '!', '?', '…'):
            → emit it (and subsequent images) immediately, each on its own block.
        - Otherwise:
            → buffer images and flush them at the NEXT sentence end.
    - At EOF:
        - if buffered images remain, optionally add terminal punctuation to the
          text (only if there is real text) and append images.
    - Image tags themselves never change sentence context (they don't affect
      last_sig / seen_real_text).
    """
    if not text.strip():
        return text

    CLOSERS = set(_CLOSERS)
    n = len(text)
    i = 0
    out_parts: List[str] = []

    tail = ""                 # last up to 2 chars of output (spacing convenience)
    last_sig: str | None = None   # last non-space, non-closer, non-image char
    seen_real_text = False        # have we seen any word-like characters?

    def append(s: str) -> None:
        """
        Append normal text (not images), updating last_sig and seen_real_text.
        """
        nonlocal tail, last_sig, seen_real_text
        if not s:
            return
        out_parts.append(s)
        for ch in s:
            if ch.isspace() or ch in CLOSERS:
                continue
            last_sig = ch
            if re.match(r"[A-Za-z0-9À-ž]", ch):
                seen_real_text = True
        tail = (tail + s)[-2:]

    def append_image(tag: str) -> None:
        """
        Append an image tag without changing sentence context.
        """
        nonlocal tail
        out_parts.append(tag)
        tail = (tail + tag)[-2:]
        # last_sig / seen_real_text intentionally untouched

    def at_block_start() -> bool:
        """
        Decide whether an image at the current position should be emitted
        immediately (block-start) or buffered.
        """
        # Start of entire document → always block start
        if not out_parts:
            return True
        # Before any real text we consider we're still in the "front image block"
        if not seen_real_text:
            return True
        # After text appears: block start only after a completed sentence
        if last_sig is None:
            return False
        return last_sig in _FINAL_PUNCT or last_sig == "…"

    img_buffer: List[str] = []

    def read_image_tag(start: int):
        m = re.match(r"<image_(\d+)>", text[start:])
        return (text[start : start + m.end()], start + m.end()) if m else (None, start)

    while i < n:
        # Image?
        if text.startswith("<image_", i):
            tag, j = read_image_tag(i)
            if tag:
                if at_block_start():
                    # Emit immediately as a block
                    if out_parts and tail != "\n\n":
                        append("\n\n")
                    append_image(tag)
                    append("\n\n")
                else:
                    img_buffer.append(tag)
                i = j
                continue

        # Normal character
        ch = text[i]
        append(ch)
        i += 1

        # Sentence end handling
        if ch in ".!?…":
            # Consume repeated end punctuation (..., !!, ??, etc.)
            while i < n and text[i] in ".!?…":
                append(text[i])
                i += 1
            # Consume trailing whitespace/closers (quotes, brackets)
            while i < n and (text[i].isspace() or text[i] in CLOSERS):
                append(text[i])
                i += 1
            # Flush buffered images after the sentence
            if img_buffer:
                append("\n\n")
                for tag in img_buffer:
                    append_image(tag)
                    append("\n\n")
                img_buffer.clear()

    result = "".join(out_parts)

    # Flush any trailing buffered images at EOF
    if img_buffer:
        result = _ensure_terminal_punctuation(result)
        result += "\n\n" + "\n\n".join(img_buffer)

    # Light whitespace normalization (keep newlines meaningful)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t\u00A0]{2,}", " ", result)
    result = re.sub(r"(^|\n)[ \t]+", r"\1", result)
    return result.strip()


# ============================================================================
# 5. Image presence / order safety checks
# ============================================================================

def _collect_all_image_tags_in_order(raw_pages: List[str]) -> List[str]:
    """
    Extract all <image_n> tags from raw pages, preserving global order.
    """
    return [t for p in raw_pages for t in re.findall(r"<image_\d+>", p)]


def _reinsert_missing_images_by_order(original_tags: List[str], text: str) -> str:
    """
    If some image tags got lost during cleaning (should not happen, but be safe),
    reinsert them near the next surviving image, or at the end if none.
    """
    if not original_tags:
        return text

    present = re.findall(r"<image_\d+>", text)
    want = Counter(original_tags)
    got = Counter(present)

    if want == got:
        return text

    for idx, tag in enumerate(original_tags):
        if got[tag] >= want[tag]:
            continue

        insert_pos = None
        for j in range(idx + 1, len(original_tags)):
            nxt = original_tags[j]
            if got[nxt] > 0:
                pos = text.find(nxt)
                if pos != -1:
                    insert_pos = pos
                    break

        if insert_pos is None:
            text = text.rstrip() + "\n\n" + tag + "\n\n"
        else:
            text = text[:insert_pos] + "\n\n" + tag + "\n\n" + text[insert_pos:]

        got[tag] += 1

    return text


def _ensure_all_images_preserved_in_order(original_tags: List[str], text: str) -> str:
    """
    Final guard: if any images are still missing, append them at the end
    in the correct global order.
    """
    want = Counter(original_tags)
    got = Counter(re.findall(r"<image_\d+>", text))

    if want == got:
        return text

    missing: List[str] = []
    for t in original_tags:
        if got[t] < want[t]:
            missing.append(t)
            got[t] += 1

    if missing:
        text = (text.rstrip() + "\n\n" + "\n\n".join(missing)).strip()

    return text


# ============================================================================
# 6. Single-page cleaning (no relocation)
# ============================================================================

def clean_page_text(
    raw: str,
    repeated_lines: Set[str] | None = None,
    remove_all_caps_headers: bool = False,
) -> str:
    """
    Clean a single page, line-by-line, without moving images globally.

    Steps:
    - Drop blank lines, repeated headers/footers, page numbers, garbage lines.
    - Optionally drop ALL CAPS headers.
    - Keep <image_n> tags as-is.
    - Dehyphenate line-break hyphens (Štem-\npiharja → Štempiharja).
    - Clean punctuation artifacts.
    - Drop standalone single-letter lines (e.g. 'a') if configured.
    - Inside a page, collapse newlines into spaces: each page becomes 1 paragraph.
    """
    if repeated_lines is None:
        repeated_lines = set()

    kept: List[str] = []

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _IMAGE_TAG_RE.fullmatch(line):
            kept.append(line)
            continue
        if line in repeated_lines:
            continue
        if _PAGE_NUMBER_RE.fullmatch(line) or _PAGE_X_RE.fullmatch(line):
            continue
        if _is_garbage_line(line):
            continue
        if remove_all_caps_headers and (
            _is_all_caps_header(line) or _ALL_CAPS_HEADER_RE.fullmatch(line)
        ):
            continue
        if _is_noisy_line(line):
            continue

        kept.append(line)

    if not kept:
        return ""

    # Newlines kept temporarily to dehyphenate across them
    text = "\n".join(kept)

    # Dehyphenate line-break hyphens (Štem-\npiharja → Štempiharja)
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Punctuation cleanup & single-letter orphan removal
    text = _clean_punctuation_artifacts(text)
    text = _drop_specific_single_letter_lines(text, DROP_SINGLE_LETTER_LINES)

    # Normalize intra-page whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t\u00A0]{2,}", " ", text)
    text = re.sub(r"(^|\n)[ \t]+", r"\1", text)

    # Flatten page lines to a single paragraph (newlines → spaces)
    text = text.replace("\n", " ")
    text = re.sub(r"[ \t\u00A0]{2,}", " ", text)

    return text.strip()


# ============================================================================
# 7. Whole-document cleaning (with global relocation)
# ============================================================================

def clean_document(pages: List[str], detect_headers: bool = True) -> str:
    """
    Clean an entire document:

    - Compute and remember all image tags and their global order.
    - Optionally detect repeated headers/footers across pages.
    - Drop nonsense/metadata/garbage pages, but keep pages containing images.
    - Clean each page individually (no relocation).
    - Concatenate pages with blank lines between them.
    - Globally relocate images forward.
    - Re-clean punctuation & single-letter orphans after relocation.
    - Ensure images are preserved in count and order, and enforce
      blank lines around each <image_n>.
    """
    if not pages:
        return ""

    original_tags = _collect_all_image_tags_in_order(pages)
    repeated: Set[str] = detect_repeated_lines(pages) if detect_headers else set()

    cleaned_pages: List[str] = []
    total = len(pages)

    for idx, page in enumerate(pages):
        stats = page_stats(page)

        if is_nonsense_page(page, stats, page_index=idx, total_pages=total):
            continue
        if is_metadata_page(page, stats, page_index=idx, total_pages=total):
            continue

        has_image = "<image_" in page
        if not has_image and is_garbage_page(page, stats):
            continue

        cleaned = clean_page_text(page, repeated_lines=repeated)
        if cleaned or has_image:
            cleaned_pages.append(cleaned)

    if not cleaned_pages:
        # Fallback: only images survived; emit them in order
        return "\n\n".join(original_tags).strip()

    # Pages separated by blank lines; each page is a single paragraph already
    doc = "\n\n".join(p for p in cleaned_pages if p is not None)

    # Global relocation and safety passes
    doc = _relocate_images_forward_globally(doc)
    doc = _clean_punctuation_artifacts(doc)
    doc = _drop_specific_single_letter_lines(doc, DROP_SINGLE_LETTER_LINES)
    doc = _reinsert_missing_images_by_order(original_tags, doc)
    doc = _force_blank_lines_around_images(doc)

    # Final whitespace normalization
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    doc = re.sub(r"[ \t\u00A0]{2,}", " ", doc)
    doc = re.sub(r"(^|\n)[ \t]+", r"\1", doc)
    doc = doc.strip()

    # Final image preservation check
    doc = _ensure_all_images_preserved_in_order(original_tags, doc)
    return doc


# ============================================================================
# 8. Convenience helpers (for marked text / analysis / nonsense pages)
# ============================================================================

def analyze_pages(pages: List[str]) -> List[Dict[str, float]]:
    """
    Return page_stats for each page, useful for debugging or tuning heuristics.
    """
    return [page_stats(p) for p in pages]


_PAGE_MARK_RE = re.compile(r"^---\s*Page\s+(\d+)\s*---\s*$", re.IGNORECASE)


def split_marked_pages(raw: str) -> List[str]:
    """
    Split a text with markers like '--- Page N ---' into per-page strings.
    """
    lines = raw.splitlines()
    pages: List[str] = []
    cur: List[str] = []

    for ln in lines:
        if _PAGE_MARK_RE.match(ln):
            if cur:
                pages.append("\n".join(cur).strip("\n"))
            cur = []
            continue
        cur.append(ln)

    if cur:
        pages.append("\n".join(cur).strip("\n"))

    return pages


def clean_marked_text(raw: str, detect_headers: bool = True) -> str:
    """
    Convenience: split a raw string by '--- Page N ---' markers and clean it.
    """
    return clean_document(split_marked_pages(raw), detect_headers=detect_headers)


def is_nonsense_page(
    text: str,
    stats: Dict[str, float],
    page_index: int,
    total_pages: int,
    edge_pages: int = 10,
    max_words: int = 80,
    max_letters_ratio: float = 0.25,
    min_garbage_line_fraction: float = 0.5,
) -> bool:
    """
    Detect “nonsense” pages in the early/late parts of the book:

    Behavior:
    - If the page contains any <image_n>, we KEEP it.
    - Only consider pages near the edges (first/last `edge_pages`).
    - Drop very short edge pages (<= 3 words) with no images (e.g. 'ap', 'IH JI').
    - Otherwise use simple text/garbage ratios and per-line garbage detection.
    """
    if "<image_" in text:
        return False

    # Only near edges (front/back)
    if not (page_index < edge_pages or page_index >= total_pages - edge_pages):
        return False

    # Super-short edge pages → treat as nonsense
    if stats["num_words"] <= 3:
        return True

    if stats["num_words"] > max_words:
        return False
    if stats["ratio_letters"] > max_letters_ratio and stats["ratio_weird"] < 0.1:
        return False

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True

    garbage = sum(1 for ln in lines if _is_garbage_line(ln))
    return (garbage / max(1, len(lines))) >= min_garbage_line_fraction
