"""PDF ingestion: extract text, detect screenshots, crop, OCR, and chunk."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pymupdf as fitz
from PIL import Image

from config import AppConfig
from ocr import OCRCache, ocr_image
from tesseract import describe, missing_languages

logger = logging.getLogger(__name__)

SERVICE_RE = re.compile(r"(?P<code>[A-Z]{2,5}-\d{3})_(?P<name>.+?)(?:\.pdf)$", re.I)


def parse_filename(path: Path) -> tuple[str, str]:
    match = SERVICE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse service code/name from filename: {path.name}")
    code = match.group("code").upper()
    name = re.sub(r"[_-]+", " ", match.group("name")).strip()
    return code, name


def normalize_name(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_manual_year(text: str) -> str:
    years = re.findall(r"\b(20\d{2}(?:[-–/]\d{2,4})?)\b", text)
    for year in years:
        if "2016" in year or "2017" in year:
            return year
    return years[0] if years else "unknown"


def block_texts(page) -> list[dict]:
    result = []
    for block in page.get_text("blocks"):
        if len(block) >= 5:
            x0, y0, x1, y1, text = block[:5]
            if text and text.strip():
                result.append({"bbox": fitz.Rect(x0, y0, x1, y1), "text": text.strip()})
    return result


def image_fingerprints(doc) -> Counter:
    counts = Counter()
    for page in doc:
        for image in page.get_images(full=True):
            xref = image[0]
            try:
                data = doc.extract_image(xref).get("image", b"")
            except Exception:
                data = b""
            if data:
                counts[hashlib.sha256(data).hexdigest()] += 1
    return counts


def merge_rectangles(rects: list[fitz.Rect], gap: float) -> list[fitz.Rect]:
    """Merge horizontally-overlapping image strips that are vertically adjacent."""
    changed = True
    rects = list(rects)
    while changed:
        changed = False
        output = []
        used = [False] * len(rects)
        for i, a in enumerate(rects):
            if used[i]:
                continue
            current = a
            used[i] = True
            merged = True
            while merged:
                merged = False
                for j, b in enumerate(rects):
                    if used[j]:
                        continue
                    horizontal_overlap = min(current.x1, b.x1) - max(current.x0, b.x0) > 0
                    vertical_gap = max(b.y0 - current.y1, current.y0 - b.y1)
                    if horizontal_overlap and vertical_gap <= gap:
                        current |= b
                        used[j] = True
                        merged = True
                        changed = True
            output.append(current)
        rects = output
    return rects


def crop_page_images(doc, page_index: int, path: Path, config: AppConfig, repeated: Counter):
    page = doc[page_index]
    page_rect = page.rect
    rects = []
    seen = set()
    for image in page.get_images(full=True):
        xref = image[0]
        try:
            extracted = doc.extract_image(xref)
            data = extracted.get("image", b"")
            ext = extracted.get("ext", "png")
            if not data:
                continue
            fp = hashlib.sha256(data).hexdigest()
            if repeated[fp] / max(1, len(doc)) > float(config.raw["pdf"]["repeated_image_page_fraction"]):
                continue
            for rect in page.get_image_rects(xref):
                if rect.get_area() < float(config.raw["pdf"]["min_image_area_points2"]):
                    continue
                rects.append(rect)
                seen.add((xref, rect.x0, rect.y0, rect.x1, rect.y1, ext))
        except Exception:
            continue

    merged = merge_rectangles(rects, float(config.raw["pdf"]["merge_vertical_gap_points"]))
    if not merged:
        return []

    dpi = float(config.raw["pdf"]["dpi"])
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    results = []
    for n, rect in enumerate(merged, start=1):
        pad = float(config.raw["pdf"]["crop_padding_points"])
        clip = fitz.Rect(
            max(page_rect.x0, rect.x0 - pad),
            max(page_rect.y0, rect.y0 - pad),
            min(page_rect.x1, rect.x1 + pad),
            min(page_rect.y1, rect.y1 + pad),
        )
        pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        out = config.crops_dir / f"{path.stem.split('_')[0].upper()}_p{page_index+1}_{n}.png"
        pix.save(str(out))
        results.append((out, clip))
    return results


def nearby_context(blocks: list[dict], rect: fitz.Rect) -> tuple[str, str]:
    above = []
    overlap = []
    for block in blocks:
        b = block["bbox"]
        if b.y1 <= rect.y0 and rect.y0 - b.y1 < 120:
            above.append(block["text"])
        if b.intersects(rect):
            overlap.append(block["text"])
    return "\n".join(above[-2:]), "\n".join(overlap)


def split_text(text: str, target: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + target)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks


def ingest_pdf(path: Path, config: AppConfig, ocr_cache: OCRCache) -> tuple[list[dict], dict]:
    service_code, service_name = parse_filename(path)
    doc = fitz.open(path)
    all_text = "\n".join(page.get_text() for page in doc)
    manual_year = find_manual_year(all_text)
    manual_title = normalize_name(next((x for x in all_text.splitlines() if "manual" in x.lower()), path.stem))

    repeated = image_fingerprints(doc)
    chunks = []
    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        blocks = block_texts(page)
        page_text = "\n".join(b["text"] for b in blocks)
        general_range = config.raw["pdf"].get("boilerplate_page_range", [1, 5])
        headings = page_text.lower()
        section = "general" if general_range[0] <= page_no <= general_range[1] or any(
            h in headings for h in ("project overview", "general information", "tools required", "starting your computer")
        ) else "service"

        crops = crop_page_images(doc, page_index, path, config, repeated)
        used_blocks = set()
        for n, (crop_path, clip) in enumerate(crops, start=1):
            step_text, callout = nearby_context(blocks, clip)
            ocr_text, confidence = ocr_image(
                crop_path, ocr_cache,
                language=config.raw["ocr"]["language"],
                psm=int(config.raw["ocr"]["psm"]),
                upscale=int(config.raw["ocr"]["upscale"]),
            )
            combined = "\n".join(x for x in (step_text, callout, ocr_text) if x).strip()
            chunk_id = f"{service_code}-p{page_no}-img{n}"
            chunks.append({
                "chunk_id": chunk_id,
                "service_code": service_code,
                "service_name": service_name,
                "manual_file": path.name,
                "manual_year": manual_year,
                "manual_title": manual_title,
                "page": page_no,
                "section": section,
                "chunk_type": "image",
                "text": combined,
                "ocr_text": ocr_text,
                "step_text": step_text,
                "image_path": str(crop_path.relative_to(config.root)),
                "ocr_confidence": confidence,
                "language_hint": "ta+en",
            })

        # Text chunks: exclude blocks that materially overlap screenshot rectangles.
        remaining = []
        for block in blocks:
            if any(block["bbox"].intersects(rect) for _, rect in crops):
                continue
            remaining.append(block["text"])
        clean = "\n".join(remaining).strip()
        target = int(config.raw["chunking"]["target_words"])
        overlap = int(config.raw["chunking"]["overlap_words"])
        for i, piece in enumerate(split_text(clean, target, overlap), start=1):
            chunks.append({
                "chunk_id": f"{service_code}-p{page_no}-txt{i}",
                "service_code": service_code,
                "service_name": service_name,
                "manual_file": path.name,
                "manual_year": manual_year,
                "manual_title": manual_title,
                "page": page_no,
                "section": section,
                "chunk_type": "text",
                "text": piece,
                "ocr_text": "",
                "step_text": "",
                "image_path": None,
                "ocr_confidence": 0.0,
                "language_hint": "en+ta",
            })
    doc.close()
    return chunks, {
        "code": service_code, "name": service_name,
        "manual_file": path.name, "manual_year": manual_year,
        "manual_title": manual_title,
    }


def ingest_all(config: AppConfig) -> tuple[list[dict], dict[str, dict]]:
    """Ingest every PDF in the manuals directory."""
    ocr_cache = OCRCache(config.cache_dir / "ocr_cache.json")
    all_chunks = []
    services = {}
    pdfs = sorted(config.manuals_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", config.manuals_dir)

    logger.info(describe())
    absent = missing_languages()
    if absent:
        logger.warning(
            "Tesseract has no language data for: %s. Screenshot text in those "
            "languages will not be extracted. See the README for install steps.",
            ", ".join(absent),
        )

    for pdf in pdfs:
        chunks, meta = ingest_pdf(pdf, config, ocr_cache)
        all_chunks.extend(chunks)
        services[meta["code"]] = meta
        logger.info("Ingested %s: %d chunks", pdf.name, len(chunks))
    return all_chunks, services
