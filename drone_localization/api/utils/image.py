"""Utility functions for API endpoints."""

import base64
import io
from typing import Tuple

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

def pixel_coords_to_jps(
    satellite_meta: dict,
    pixel_coords: Tuple[float, float],
    img_width: int = 768,
    img_height: int = 768
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to GPS coordinates (latitude, longitude).

    Args:
        satellite_metadata (dict): Dictionary containing satellite image metadata with keys:
            - tl_E: longitude of top-left corner
            - tl_N: latitude of top-left corner
            - br_E: longitude of bottom-right corner
            - br_N: latitude of bottom-right corner
        pixel_coords (tuple): Pixel coordinates (x, y) on the satellite image.
        img_width (int): Width of the satellite image in pixels.
        img_height (int): Height of the satellite image in pixels.

    Returns:
        tuple: GPS coordinates (latitude, longitude).

    Note:
        Our test images are square 768x768 pixels, but for non-test datasets
        you need to provide the actual width and height of the satellite image.
    """
    tl_N = satellite_metadata["tl_N"]  # latitude top-left
    tl_E = satellite_metadata["tl_E"]  # longitude top-left
    br_N = satellite_metadata["br_N"]  # latitude bottom-right
    br_E = satellite_metadata["br_E"]  # longitude bottom-right

    lat_range = tl_N - br_N
    lon_range = br_E - tl_E

    lat_per_pixel = lat_range / img_height
    lon_per_pixel = lon_range / img_width

    jps_lon = tl_E + pixel_coords[0] * lon_per_pixel
    jps_lat = tl_N - pixel_coords[1] * lat_per_pixel

    return (jps_lat, jps_lon)