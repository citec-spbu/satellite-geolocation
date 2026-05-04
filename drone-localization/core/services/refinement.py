from typing import Tuple

class RefinementService:
    def __init__(self, model_path: str = "models/refinement.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        # TODO: self.model = torch.load(self.model_path, map_location="cpu")
        self.model = "loaded"
        print("✅ Модель уточнения загружена")

    def calculate_position(self, drone_image_b64: str, satellite_image_b64: str) -> Tuple[float, float]:
        if self.model is None:
            self.load_model()

        # TODO:
        # 1. Подготовь тензоры пары изображений
        # 2. output = self.model(drone_tensor, sat_tensor)
        # 3. Распакуй предсказанные (lat, lon) из output

        # Заглушка:
        return 55.7558, 37.6173  # lat, lon