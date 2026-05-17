import io
import logging
import uuid
from typing import List, Tuple

import numpy as np
from minio import Minio
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from ..configs.settings import get_settings
from ..core.interfaces.gallery_repository import GalleryRepository

logger = logging.getLogger(__name__)


class GalleryRepositoryImpl(GalleryRepository):
    def __init__(self):
        settings = get_settings()
        self.qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection = settings.qdrant_collection
        self.minio = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False,
        )
        self.bucket = settings.minio_bucket
        # Убедимся, что бакет существует
        if not self.minio.bucket_exists(self.bucket):
            self.minio.make_bucket(self.bucket)
        # Создадим коллекцию, если её нет
        self._init_qdrant_collection()

    def _init_qdrant_collection(self):
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=qdrant_models.VectorParams(
                    size=512, distance=qdrant_models.Distance.COSINE
                ),
            )
            logger.info(f"Created Qdrant collection '{self.collection}'")

    def search_similar(
        self, embedding: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=embedding.tolist(),
            limit=top_k,
        )
        return [(hit.payload["image_id"], hit.score) for hit in results]

    def get_image(self, image_id: str) -> Image.Image:
        response = self.minio.get_object(self.bucket, image_id)
        return Image.open(io.BytesIO(response.read()))

    def add_image(
        self, image_bytes: bytes, embedding: np.ndarray, metadata: dict
    ) -> str:
        image_id = str(uuid.uuid4())
        # Сохраняем в MinIO
        self.minio.put_object(
            self.bucket,
            image_id,
            io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=metadata.get("content_type", "image/jpeg"),
        )
        # Вставляем в Qdrant
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[
                qdrant_models.PointStruct(
                    id=image_id,
                    vector=embedding.tolist(),
                    payload={**metadata, "image_id": image_id},
                )
            ],
        )
        logger.info(f"Added image {image_id} to gallery")
        return image_id

    def count(self) -> int:
        try:
            return self.qdrant.count(collection_name=self.collection).count
        except Exception:
            return 0

    def delete_image(self, image_id: str) -> bool:
        try:
            # Удаляем из MinIO
            self.minio.remove_object(self.bucket, image_id)
            # Удаляем из Qdrant
            self.qdrant.delete(
                collection_name=self.collection,
                points_selector=qdrant_models.PointIdsList(points=[image_id]),
            )
            logger.info(f"Deleted image {image_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete image {image_id}: {e}")
            return False

    def clear(self) -> None:
        # Очищаем Qdrant коллекцию
        self.qdrant.delete_collection(collection_name=self.collection)
        self._init_qdrant_collection()
        # Очищаем MinIO бакет (удаляем все объекты)
        objects = list(self.minio.list_objects(self.bucket))
        for obj in objects:
            self.minio.remove_object(self.bucket, obj.object_name)
        logger.info("Gallery cleared")

    def health_check(self) -> bool:
        try:
            # Проверяем MinIO
            self.minio.bucket_exists(self.bucket)
            # Проверяем Qdrant
            self.qdrant.get_collection(collection_name=self.collection)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
