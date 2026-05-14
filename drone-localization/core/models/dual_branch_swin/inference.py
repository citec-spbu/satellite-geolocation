import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


class DroneToSatelliteRetrieval:
    def __init__(self, weights_path, embeddings_path, device=None):
       self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Загружаем модель
        from .model import DualBranchSwin
        self.model = DualBranchSwin().to(self.device)
        self._load_weights(weights_path)
        self.model.eval()
        
        # Загружаем базу спутниковых эмбеддингов
        data = np.load(embeddings_path)
        self.sat_embeddings = data['embeddings']  
        self.sat_paths = data['paths']            
        
        # Трансформации
        from .dataset import get_transforms
        self.transform = get_transforms(is_train=False)
    
    def _load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Убираем префикс module. если есть
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
    
    def find(self, drone_image_path, top_k=1):
        # Загружаем изображение
        if isinstance(drone_image_path, str):
            img = Image.open(drone_image_path).convert('RGB')
        else:
            img = drone_image_path  # уже PIL Image
        
        # Преобразуем
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        is_drone = torch.tensor([True], device=self.device)
        
        # Извлекаем эмбеддинг
        with torch.no_grad():
            drone_emb = self.model(img_tensor, is_drone)
        drone_emb = drone_emb.cpu().numpy().flatten()
        
        # Косинусное сходство
        similarities = np.dot(self.sat_embeddings, drone_emb)
        
        if top_k == 1:
            best_idx = np.argmax(similarities)
            return str(self.sat_paths[best_idx])
        
        # Топ-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(str(self.sat_paths[i]), float(similarities[i])) for i in top_indices]
