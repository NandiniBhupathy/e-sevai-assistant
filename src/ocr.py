"""OCR utilities with image-hash caching."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from tesseract import configure_tesseract

logger = logging.getLogger(__name__)


def image_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class OCRCache:
    """Persistent OCR cache keyed by image SHA-256."""

    def __init__(self, cache_path: str | Path):
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except json.JSONDecodeError:
            self.data = {}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value: dict):
        self.data[key] = value
        self.save()


def ocr_image(image_path: str | Path, cache: OCRCache, language: str = "eng+tam",
              psm: int = 6, upscale: int = 2) -> tuple[str, float]:
    """OCR an image, returning text and mean confidence."""
    configure_tesseract()
    path = Path(image_path)
    raw = path.read_bytes()
    key = image_hash_bytes(raw)
    cached = cache.get(key)
    if cached:
        return cached.get("text", ""), float(cached.get("confidence", 0.0))

    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image)
    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale))
    try:
        data = pytesseract.image_to_data(
            image,
            lang=language,
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )
        texts = []
        confidences = []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            try:
                value = float(conf)
            except (ValueError, TypeError):
                value = -1
            if text:
                texts.append(text)
                if value >= 0:
                    confidences.append(value)
        text = " ".join(texts).strip()
        confidence = float(np.mean(confidences)) if confidences else 0.0
    except Exception as exc:
        logger.exception("OCR failed for %s: %s", path, exc)
        text, confidence = "", 0.0

    cache.put(key, {"text": text, "confidence": confidence})
    return text, confidence
