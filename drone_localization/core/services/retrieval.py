import base64
import io
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from drone_localization.core.models.ConvNext import two_view_net
from drone_localization.core.utils.utils_for_retrieval import load_network

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    image: str  # base64 спутникового снимка
    score: float  # похожесть (0.0 - 1.0)
    path: str  # путь к найденному файлу (для отладки)


class RetrievalService:
    """Сервис поиска спутниковых изображений по фото с дрона"""

    def __init__(
        self,
        model_name: str = "convnext_tri",
        data_dir: str = "data",  # ← просто "data", на одном уровне с core/
        dataset_path: str = "/content/SUES-200-512x512",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            model_name: имя модели (подпапка в data/)
            data_dir: путь к папке с данными моделей
            dataset_path: путь к датасету
            device: устройство для инференса
            image_size: размер входных изображений
        """
        self.device = device
        self.image_size = image_size

        # Пути
        model_dir = os.path.join(data_dir, model_name)
        self.weights_dir = os.path.join(model_dir, "weights")
        self.gallery_dir = os.path.join(model_dir, "gallery")
        self.dataset_path = dataset_path

        # Загружаемые компоненты
        self.model = None
        self.transform = None
        self.gallery_embeddings = None
        self.gallery_paths = None

        logger.info(f"RetrievalService initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Weights: {self.weights_dir}")
        logger.info(f"  Gallery: {self.gallery_dir}")
        logger.info(f"  Dataset: {self.dataset_path}")
        logger.info(f"  Device: {self.device}")

    def load_model(self):
        """Загрузка модели из весов"""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info("Loading model...")

        # Создаем конфиг для загрузки
        class Opt:
            def __init__(self):
                self.name = "convnext_tri"
                self.views = 2
                self.block = 2
                self.share = True
                self.nclasses = 200
                self.resnet = False
                self.fp16 = False
                self.train_all = True
                self.droprate = 0.5
                self.color_jitter = True
                self.batchsize = 8
                self.h = 256
                self.w = 256
                self.erasing_p = 0.5
                self.lr = 0.01

        opt = Opt()

        model, _, epoch = load_network(name=opt.name, weights_dir=self.weights_dir)

        self.model = model.to(self.device)
        self.model.eval()

        # Трансформации
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.image_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        logger.info(f"Model loaded (epoch {epoch})")

    def load_or_build_gallery(self, num_buildings: int = 40):
        """Загрузка галереи из кэша или построение новой"""
        if self.model is None:
            raise RuntimeError("Model must be loaded first")

        os.makedirs(self.gallery_dir, exist_ok=True)

        embeddings_cache = os.path.join(self.gallery_dir, "embeddings.npy")
        paths_cache = os.path.join(self.gallery_dir, "paths.json")

        if os.path.exists(embeddings_cache) and os.path.exists(paths_cache):
            logger.info("Loading gallery from cache...")
            self.gallery_embeddings = np.load(embeddings_cache)
            with open(paths_cache, "r") as f:
                self.gallery_paths = json.load(f)
            logger.info(f"Gallery loaded: {len(self.gallery_embeddings)} embeddings")
        else:
            logger.info(f"Building gallery from first {num_buildings} buildings...")
            self.gallery_embeddings, self.gallery_paths = self._build_gallery(
                num_buildings
            )

            np.save(embeddings_cache, self.gallery_embeddings)
            with open(paths_cache, "w") as f:
                json.dump(self.gallery_paths, f)
            logger.info("Gallery cached")

    def find_match(self, drone_image_b64: str) -> RetrievalResult:
        """
        Поиск наиболее похожего спутникового снимка

        Args:
            drone_image_b64: base64 изображения с дрона

        Returns:
            RetrievalResult с base64 спутника и score
        """
        if self.model is None:
            self.load_model()

        if self.gallery_embeddings is None:
            self.load_or_build_gallery()

        # Декодируем и обрабатываем изображение с дрона
        drone_tensor = self._decode_base64_to_tensor(drone_image_b64)

        # Получаем эмбеддинг дрона
        drone_embedding = self._get_drone_embedding(drone_tensor)

        # Вычисляем косинусное сходство со всей галереей
        similarities = np.dot(self.gallery_embeddings, drone_embedding)

        # Находим индекс лучшего совпадения
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        # Загружаем спутниковое изображение и кодируем в base64
        sat_image_path = self.gallery_paths[best_idx]
        with open(sat_image_path, "rb") as f:
            sat_image_bytes = f.read()
        sat_image_b64 = base64.b64encode(sat_image_bytes).decode("utf-8")

        return RetrievalResult(
            image=sat_image_b64, score=best_score, path=sat_image_path
        )

    def find_top_k(self, drone_image_b64: str, top_k: int = 5) -> List[Dict]:
        """
        Поиск top-k спутниковых снимков (для отладки)

        Args:
            drone_image_b64: base64 изображения с дрона
            top_k: количество результатов

        Returns:
            Список словарей с 'image', 'score', 'path'
        """
        if self.model is None:
            self.load_model()

        if self.gallery_embeddings is None:
            self.load_or_build_gallery()

        drone_tensor = self._decode_base64_to_tensor(drone_image_b64)
        drone_embedding = self._get_drone_embedding(drone_tensor)

        similarities = np.dot(self.gallery_embeddings, drone_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sat_image_path = self.gallery_paths[idx]
            with open(sat_image_path, "rb") as f:
                sat_image_bytes = f.read()
            sat_image_b64 = base64.b64encode(sat_image_bytes).decode("utf-8")

            results.append(
                {
                    "image": sat_image_b64,
                    "score": float(similarities[idx]),
                    "path": sat_image_path,
                }
            )

        return results

    def _decode_base64_to_tensor(self, base64_string: str) -> torch.Tensor:
        """Декодирование base64 строки в тензор"""
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        return image_tensor

    def _get_drone_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Извлечение эмбеддинга для изображения с дрона"""
        with torch.no_grad():
            # drone view: model(None, drone_image)
            _, features = self.model(None, image_tensor)

            # features может быть кортежем (tensor, embedding) или просто тензором
            if isinstance(features, tuple):
                embedding = features[1]  # основной эмбеддинг [B, 512]
            else:
                embedding = features

            # Нормализуем для косинусного сходства
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().flatten()

    def _get_satellite_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Извлечение эмбеддинга для спутникового изображения"""
        with torch.no_grad():
            # satellite view: model(satellite_image, None)
            features, _ = self.model(image_tensor, None)

            if isinstance(features, tuple):
                embedding = features[1]
            else:
                embedding = features

            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().flatten()

    def _build_gallery(self, num_buildings: int) -> Tuple[np.ndarray, List[str]]:
        """Построение галереи спутниковых эмбеддингов"""
        satellite_dir = os.path.join(self.dataset_path, "satellite-view")

        if not os.path.exists(satellite_dir):
            raise FileNotFoundError(f"Satellite directory not found: {satellite_dir}")

        embeddings_list = []
        paths_list = []

        # Получаем список зданий и берем первые num_buildings
        location_ids = sorted(os.listdir(satellite_dir))[:num_buildings]

        logger.info(f"Processing {len(location_ids)} buildings...")

        for location_id in location_ids:
            location_path = os.path.join(satellite_dir, location_id)

            if not os.path.isdir(location_path):
                continue

            # Берем все изображения в папке здания
            for img_name in sorted(os.listdir(location_path)):
                if not img_name.endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(location_path, img_name)

                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)

                    embedding = self._get_satellite_embedding(image_tensor)

                    embeddings_list.append(embedding)
                    paths_list.append(img_path)
                except Exception as e:
                    logger.info(f"  Warning: Failed to process {img_path}: {e}")

        gallery_embeddings = np.stack(embeddings_list)
        logger.info(
            f"Gallery built: {len(gallery_embeddings)} embeddings from {len(location_ids)} buildings"
        )

        return gallery_embeddings, paths_list
