from dataclasses import dataclass
from utils.retrieval_helpers import base64_to_tensor

@dataclass
class RetrievalResult:
    image: str   # base64 спутникового снимка
    score: float # похожесть (0.0 - 1.0)

class RetrievalService:
    def __init__(self, model_path: str = "models/retrieval.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        # TODO: self.model = torch.load(self.model_path, map_location="cpu")
        self.model = "loaded"  # заглушка
        print("✅ Модель поиска загружена")

    def find_match(self, drone_image_b64: str) -> RetrievalResult:
        if self.model is None:
            self.load_model()

        # TODO:
        # 1. Декодируй base64 -> numpy/PIL
        # 2. embeddings = self.model(drone_tensor)
        # 3. Найди ближайший в базе (FAISS / простой поиск)
        # 4. Загрузи картинку спутника, переведи в base64

        # 1.
        drone_tensor = base64_to_tensor(drone_image_b64)
        
        # 2.
        embeddings = self.model(drone_tensor)


        # Сейчас возвращаем заглушку для проверки API:
        VALID_BLANK_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        return RetrievalResult(image=VALID_BLANK_IMAGE, score=0.94)