import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torchvision import transforms


def base64_to_image(b64_str: str) -> Image.Image:
    """Конвертирует base64 строку в PIL.Image"""
    try:
        # Иногда base64 строки содержат префикс data:image/png;base64,
        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Конвертирует PIL.Image в base64 строку"""
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_tensor(b64_str: str) -> torch.Tensor:
    """Конвертирует base64 в torch.Tensor для модели"""
    image = base64_to_image(b64_str)
    
    # Стандартные трансформы для ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image).unsqueeze(0)  # [1, 3, 224, 224]


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Конвертирует torch.Tensor обратно в base64 (для отладки)"""
    # Обратная нормализация
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    
    image = Image.fromarray(tensor)
    return image_to_base64(image)