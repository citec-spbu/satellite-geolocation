"""Utility functions for API endpoints."""

import base64
import io

from PIL import Image


def base64_to_pil(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def pil_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")