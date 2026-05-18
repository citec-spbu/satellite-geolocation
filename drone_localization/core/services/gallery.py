import io
import logging
import os
from typing import List, Tuple
import json

import numpy as np
from PIL import Image

from ..interfaces.gallery_repository import GalleryRepository
from .inference import InferenceService

logger = logging.getLogger(__name__)


class GalleryService:
    def __init__(self, inference: InferenceService, repository: GalleryRepository):
        self.inference = inference
        self.repo = repository

    def upload_image(self, image: Image.Image, metadata: dict = None) -> str:
        """Загружает одиночное спутниковое изображение в галерею."""
        emb = self.inference.extract_satellite_embedding(image)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        if metadata is None:
            metadata = {}
        metadata.setdefault("content_type", "image/jpeg")
        return self.repo.add_image(image_bytes, emb, metadata)

    def get_image(self, image_id: str) -> Image.Image:
        return self.repo.get_image(image_id)

    def search_similar(
        self, embedding: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        return self.repo.search_similar(embedding, top_k)

    def build_from_dataset(self, dataset_path: str, num_buildings: int = 40) -> int:
        """Загружает спутниковые снимки из датасета (для первоначального наполнения)."""
        satellite_dir = os.path.join(dataset_path, "satellite-view")
        if not os.path.exists(satellite_dir):
            raise FileNotFoundError(f"Satellite directory not found: {satellite_dir}")

        location_ids = sorted(os.listdir(satellite_dir))[:num_buildings]
        uploaded = 0
        for location_id in location_ids:
            loc_path = os.path.join(satellite_dir, location_id)
            if not os.path.isdir(loc_path):
                continue
            for img_name in sorted(os.listdir(loc_path)):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(loc_path, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    self.upload_image(
                        img, metadata={"filename": img_name, "location": location_id}
                    )
                    uploaded += 1
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
        logger.info(f"Built gallery from dataset: {uploaded} images")
        return uploaded
