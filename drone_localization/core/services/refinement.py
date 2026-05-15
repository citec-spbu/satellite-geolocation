import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from drone_localization.core.models.utils_for_refinement.configuration_file import (
    config,
)
from drone_localization.core.models.utils_for_refinement.taskflow import FPI

logger = logging.getLogger(__name__)


class RefinementService:
    def __init__(
        self,
        model_path: str = "models/refinement.pt",
        config_path: str = "models/new_example_config.json",
    ):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Разрешаем путь к конфигурации относительно текущего файла

        # Если что это данил добавил, у меня на компе почему то не хотел находить в том виде как у тебя было написано
        # изза этого сервак не запускался
        config_path = Path(__file__).parent / config_path
        self.opt = config(str(config_path))

    def load_model(self):
        """Загружает модель regressor"""
        if self.model is None:
            self.model = FPI(self.opt, device=self.device)
            model_keys = set(self.model.state_dict().keys())
            ckpt = torch.load(self.model_path, map_location=self.device)
            new_state_dict = {}
            for k, v in ckpt.items():
                if k in model_keys:
                    new_state_dict[k] = v
                elif "backbone_uav.backbone." + k in model_keys:
                    new_state_dict["backbone_uav.backbone." + k] = v
            missing, unexpected = self.model.load_state_dict(
                new_state_dict, strict=True
            )
            self.model.to(self.device)
            logger.info(
                f"✅ Refinement model loaded on {self.device} with {len(missing)} missing and {len(unexpected)} unexpected keys"
            )  # для отладки

    def _get_transformer(self):
        transform_uav_list = [
            transforms.Resize(self.opt.data_config["UAVhw"], interpolation=3),
            transforms.ToTensor(),
        ]

        transform_satellite_list = [
            transforms.Resize(self.opt.data_config["Satellitehw"], interpolation=3),
            transforms.ToTensor(),
        ]

        data_transforms = {
            "UAV": transforms.Compose(transform_uav_list),
            "satellite": transforms.Compose(transform_satellite_list),
        }
        return data_transforms

    def calculate_position(
        self, drone_image: Image, satellite_image: Image
    ) -> Tuple[float, float]:
        """
        Уточняет координаты по паре изображений

        Args:
            drone_image: PIL Image object for drone image
            satellite_image: PIL Image object for satellite image

        Returns:
            Tuple (x, y) - уточненные координаты на изображении спутника
        """
        transformation = self._get_transformer()
        if self.model is None:
            self.load_model()
        # 2. Декодирование строки в бинарные данные
        with Image.open(drone_image) as img:
            x = transformation["UAV"](img).unsqueeze(0).to(self.device)
        with Image.open(satellite_image) as img:
            z = transformation["satellite"](img).unsqueeze(0).to(self.device)
            height = img.height
            width = img.width
        self.model.eval()
        with torch.no_grad():
            cls_out, reg_out = self.model(x, z)  # используем только сls reg пустой
        normalized = torch.sigmoid(cls_out[0]).squeeze().detach().numpy()
        kernel = self.create_hanning_mask(self.opt.test_config["filterR"])
        map = cv2.filter2D(normalized, -1, kernel)
        pos = np.argmax(map)
        coords = np.unravel_index(pos, map.shape)
        x_original = (coords[0] / 576) * height
        y_original = (coords[1] / 576) * width
        return x_original, y_original

    @staticmethod
    def create_hanning_mask(center_R):
        hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
            np.hanning(center_R + 2), np.hanning(center_R + 2)
        )
        hann_window /= hann_window.sum()
        return hann_window[1:-1, 1:-1]
