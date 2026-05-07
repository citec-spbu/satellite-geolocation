from dataclasses import dataclass
from typing import Optional
from core.utils.image import base64_to_tensor
import torch

@dataclass
class RetrievalResult:
    image: str                    # base64 спутникового снимка
    score: float                  # похожесть (0.0 - 1.0)
    tile_id: Optional[str] = None # ID тайла для отладки
    coordinates: Optional[dict] = None  # {"lat": ..., "lon": ...} из метаданных

class RetrievalService:
    def __init__(self, model_path: str = "models/retrieval.pt"):  # ✅ __init__!
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Загружает модель encoder"""
        if self.model is None:
            # TODO: когда Саша сделает модель:
            # self.model = torch.load(self.model_path, map_location=self.device)
            self.model = "mock_encoder"  # заглушка
            print(f"✅ Retrieval model loaded on {self.device}")
    
    def find_match(self, drone_image_b64: str) -> RetrievalResult:
        """
        Находит ближайший спутниковый тайл
        
        Args:
            drone_image_b64: Base64 изображение с дрона
        
        Returns:
            RetrievalResult с лучшим совпадением
        """
        if self.model is None:
            self.load_model()

        # TODO: РЕАЛЬНАЯ ЛОГИКА:
        # 1. tensor = base64_to_tensor(drone_image_b64)
        # 2. embedding = self.model(tensor)
        # 3. distances, indices = faiss_index.search(embedding, top_k)
        # 4. Загрузить satellite image из хранилища по index
        # 5. Загрузить metadata (tile_id, coordinates)
        
        # ЗАГЛУШКА для тестов:
        VALID_BLANK_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        return RetrievalResult(
            image=VALID_BLANK_IMAGE,
            score=0.8,
            tile_id="mock_tile_001",
            coordinates={"lat": 59.9343, "lon": 30.3351}  # Санкт-Петербург
        )