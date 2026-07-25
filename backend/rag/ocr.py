"""OCR and image processing for multimodal RAG."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}


def extract_text_from_image(image: Image.Image) -> str:
    if not TESSERACT_AVAILABLE:
        return "[OCR unavailable: install Tesseract OCR on your system]"
    try:
        text = pytesseract.image_to_string(image)
        return text.strip() or "[No text detected in image]"
    except Exception as exc:
        return f"[OCR error: {exc}]"


def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_image_from_base64(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return load_image_from_bytes(base64.b64decode(b64))


def save_uploaded_image(data: bytes, uploads_dir: Path, filename: str) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / filename
    dest.write_bytes(data)
    return dest


def process_image_bytes(data: bytes) -> dict[str, str]:
    image = load_image_from_bytes(data)
    ocr_text = extract_text_from_image(image)
    return {
        "ocr_text": ocr_text,
        "width": str(image.width),
        "height": str(image.height),
    }


def process_image_base64(b64: str) -> dict[str, str]:
    image = load_image_from_base64(b64)
    ocr_text = extract_text_from_image(image)
    return {
        "ocr_text": ocr_text,
        "width": str(image.width),
        "height": str(image.height),
    }
