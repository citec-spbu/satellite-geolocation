import io
import json
import logging
import os
from typing import List, Tuple

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
        logger.info('try to make embeddings')
        emb = self.inference.extract_satellite_embedding(image)
        logger.info('embedding made')
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

    def import_dataset_with_metadata(self, dataset_path: str, max_images: int = None) -> int:
        """
        Импортирует датасет в формате:
        data/
          └── location_name/
              ├── satellite.jpg (или .png)
              ├── uav.jpg (или .png)
              └── metadata.json

        Args:
            dataset_path: Путь к корневой папке датасета
            max_images: Максимальное количество изображений для импорта (None = все)

        Returns:
            Количество успешно импортированных изображений
        """
        uploaded = 0
        skipped = 0
        errors = 0

        # Проходим по всем локациям
        for location_name in os.listdir(dataset_path):
            location_path = os.path.join(dataset_path, location_name)
            if not os.path.isdir(location_path):
                continue

            metadata_file = os.path.join(location_path, "metadata.json")

            # Загружаем метаданные если есть
            metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
                    metadata = {}

            # Ищем файл satellite (с разными расширениями)
            satellite_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(location_path, f"satellite{ext}")
                if os.path.exists(candidate):
                    satellite_file = candidate
                    break

            if not satellite_file:
                logger.warning(f"No satellite image found in {location_path}, skipping")
                skipped += 1
                continue

            if max_images and uploaded >= max_images:
                break

            try:
                img = Image.open(satellite_file).convert("RGB")

                # Формируем метаданные для загрузки
                upload_metadata = {
                    "filename": os.path.basename(satellite_file),
                    "location": location_name,
                    "dataset_format": "uav_satellite_pair"
                }

                # Добавляем информацию из metadata.json если есть
                if metadata:
                    # Добавляем координаты из satellite секции
                    if "satellite" in metadata:
                        sat_meta = metadata["satellite"]
                        if all(k in sat_meta for k in ["tl_E", "tl_N", "br_E", "br_N"]):
                            logger.info("try to give coordinated")
                            upload_metadata["coordinates"] = {
                                "tl_E": sat_meta["tl_E"],
                                "tl_N": sat_meta["tl_N"],
                                "br_E": sat_meta["br_E"],
                                "br_N": sat_meta["br_N"]
                            }
                            logger.info('coordinates was given')
                    # Добавляем GPS координаты UAV
                    if "uav_gps" in metadata:
                        upload_metadata["uav_gps"] = metadata["uav_gps"]

                    # Добавляем высоту UAV
                    if "uav_height_m" in metadata:
                        upload_metadata["uav_height_m"] = metadata["uav_height_m"]

                    # Добавляем ID объекта
                    if "object_id" in metadata:
                        upload_metadata["object_id"] = metadata["object_id"]

                logger.info('try to upload image')
                # Загружаем изображение
                self.upload_image(img, upload_metadata)
                uploaded += 1
                logger.info('image is uploaded')

            except Exception as e:
                logger.error(f"Failed to process {satellite_file}: {e}")
                errors += 1

        logger.info(f"Dataset import completed: {uploaded} uploaded, {skipped} skipped, {errors} errors")
        return uploaded