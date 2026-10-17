"""Show whether key screenshot-only facts occur in PDF text vs OCR."""
from __future__ import annotations

import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from ocr import OCRCache, ocr_image
from ingest import crop_page_images, image_fingerprints


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/check_text_gap.py path/to/manual.pdf")
    path = Path(sys.argv[1]).resolve()
    config = load_config(ROOT / "config.yaml")
    doc = fitz.open(path)
    raw = "\n".join(p.get_text() for p in doc).lower()
    repeated = image_fingerprints(doc)
    cache = OCRCache(config.cache_dir / "ocr_cache.json")
    ocr = []
    for i in range(len(doc)):
        for crop, _ in crop_page_images(doc, i, path, config, repeated):
            text, _ = ocr_image(crop, cache, config.raw["ocr"]["language"], int(config.raw["ocr"]["psm"]), int(config.raw["ocr"]["upscale"]))
            ocr.append(text.lower())
    ocr_text = "\n".join(ocr)
    for needle in ("chitta", "adangal", "200 kb"):
        print(f"{needle:10} raw_text={'YES' if needle in raw else 'NO'}  ocr={'YES' if needle in ocr_text else 'NO'}")
    doc.close()


if __name__ == "__main__":
    main()
