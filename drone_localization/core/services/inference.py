import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path

from ..models.ConvNext.retrieval_utils import load_network

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        model_name="convnext_tri",
        weights_dir="weights",
        device="cuda",
        image_size=(256, 256),
        model_weights="net_152.pth"
    ):
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.image_size = image_size
        self.model_weights = model_weights

        class Opt:
            def __init__(self):
                self.name = model_name
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
                self.h = image_size[0]
                self.w = image_size[1]
                self.erasing_p = 0.5
                self.lr = 0.01

        opt = Opt()

        self.model, _, _ = load_network(name=opt.name, opt=opt, weights_dir=weights_dir, model_weights=model_weights)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.image_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        logger.info(f"InferenceService loaded on {self.device}")

    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def extract_drone_embedding(self, drone_img: Image.Image) -> np.ndarray:
        tensor = self._preprocess(drone_img)
        with torch.no_grad():
            _, features = self.model(None, tensor)
        return self._extract_embedding(features)

    def extract_satellite_embedding(self, sat_img: Image.Image) -> np.ndarray:
        tensor = self._preprocess(sat_img)
        with torch.no_grad():
            features, _ = self.model(tensor, None)
        return self._extract_embedding(features)

    def _extract_embedding(self, features) -> np.ndarray:
        if isinstance(features, tuple):
            emb = features[1]  # основной эмбеддинг
        else:
            emb = features
        emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().flatten()
