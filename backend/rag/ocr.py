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

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    _paddle_instance = None
except ImportError:
    PADDLE_AVAILABLE = False
    _paddle_instance = None


SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}


def extract_text_from_image(image: Image.Image) -> str:
    # 1. Try PaddleOCR for highest table & document accuracy if installed
    if PADDLE_AVAILABLE:
        try:
            global _paddle_instance
            if _paddle_instance is None:
                _paddle_instance = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            import numpy as np
            img_np = np.array(image)
            result = _paddle_instance.ocr(img_np, cls=True)
            lines = []
            if result and result[0]:
                for line in result[0]:
                    lines.append(line[1][0])
            if lines:
                return "\n".join(lines).strip()
        except Exception:
            pass

    # 2. Fallback to Tesseract OCR
    if TESSERACT_AVAILABLE:
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception:
            pass

    return ""


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
