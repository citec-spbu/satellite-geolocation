import base64
import io
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
from .inference import InferenceService
from .gallery import GalleryService
from ..schemas.retrieval import RetrievalResult
from infrastructure.gallery_repository_impl import GalleryRepositoryImpl
logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Сервис поиска спутниковых изображений по фото с дрона.
    Сохраняет обратную совместимость с API, принимая base64 строки.
    Внутренняя логика работает с PIL.Image.
    """
    def __init__(
        self,
        dataset_path: str = "/content/SUES-200-512x512",
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.dataset_path = dataset_path
        # Инициализируем инфраструктурные компоненты
        self.inference = InferenceService(device=device, image_size=image_size)
        self.gallery_repo = GalleryRepositoryImpl()
        self.gallery = GalleryService(self.inference, self.gallery_repo)
        # Ленивая загрузка галереи
        self._gallery_loaded = False

    def _ensure_gallery(self):
        if not self._gallery_loaded:
            if self.gallery_repo.count() == 0:
                logger.info("Gallery is empty, building from dataset...")
                self.gallery.build_from_dataset(self.dataset_path)
            self._gallery_loaded = True

    # Старые публичные методы для совместимости с api/localize.py
    def find_match(self, drone_image_b64: str) -> RetrievalResult:
        self._ensure_gallery()
        drone_img = self._b64_to_pil(drone_image_b64)
        return self._find_match_pil(drone_img)

    def find_top_k(self, drone_image_b64: str, top_k: int = 5) -> List[Dict]:
        """Для совместимости возвращает список словарей (старый формат)."""
        self._ensure_gallery()
        drone_img = self._b64_to_pil(drone_image_b64)
        results = self._find_top_k_pil(drone_img, top_k)
        # Конвертируем PIL.Image в base64 для старых потребителей
        return [
            {
                "image": self._pil_to_b64(res.image),
                "score": res.score,
                "path": getattr(res, 'path', '')  # path больше не хранится
            }
            for res in results
        ]

    # Новые методы для работы с PIL.Image (будущее API)
    def find_match_pil(self, drone_img: Image.Image) -> RetrievalResult:
        self._ensure_gallery()
        return self._find_match_pil(drone_img)

    def find_top_k_pil(self, drone_img: Image.Image, top_k: int = 5) -> List[RetrievalResult]:
        self._ensure_gallery()
        return self._find_top_k_pil(drone_img, top_k)

    # Внутренняя реализация
    def _find_match_pil(self, drone_img: Image.Image) -> RetrievalResult:
        emb = self.inference.extract_drone_embedding(drone_img)
        hits = self.gallery.search_similar(emb, top_k=1)
        if not hits:
            raise RuntimeError("Gallery is empty")
        best_id, best_score = hits[0]
        sat_img = self.gallery.get_image(best_id)
        return RetrievalResult(image=sat_img, score=best_score)

    def _find_top_k_pil(self, drone_img: Image.Image, top_k: int) -> List[RetrievalResult]:
        emb = self.inference.extract_drone_embedding(drone_img)
        hits = self.gallery.search_similar(emb, top_k=top_k)
        results = []
        for img_id, score in hits:
            sat_img = self.gallery.get_image(img_id)
            results.append(RetrievalResult(image=sat_img, score=score))
        return results

    # Вспомогательные методы для конвертации base64 <-> PIL
    @staticmethod
    def _b64_to_pil(b64_str: str) -> Image.Image:
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]
        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    @staticmethod
    def _pil_to_b64(img: Image.Image, format="JPEG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")