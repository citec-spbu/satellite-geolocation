import base64
import torch
from io import BytesIO
from PIL import Image
from torchvision.transforms import ToTensor

def base64_to_tensor(base64_string):
    # 1. Декодируем base64 в байты
    image_bytes = base64.b64decode(base64_string)
    
    # 2. Открываем изображение с помощью PIL
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # 3. Преобразуем PIL изображение в тензор (значения от 0 до 1)
    transform = ToTensor()
    tensor = transform(image)
    
    return tensor
