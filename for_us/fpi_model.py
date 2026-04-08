"""FPI inference module for external use.

This module wraps the DRL FPI model so it can be imported and used in
another project with two input images.

Usage example:
    from for_us.fpi_model import FPIInference
    inf = FPIInference(config_path, checkpoint_path, device='cuda')
    result = inf.predict(uav_image, satellite_image)
    print(result['pixel_coordinate'])
"""

import pathlib
import os
import sys
from typing import Tuple, Union, Dict
from distill.configuration_file import config
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ensure repository root is importable when this module is used from another project.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from distill.taskflow import make_model


def _to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image.astype('uint8')).convert('RGB')
        if image.ndim == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = (255 * np.clip(image, 0.0, 1.0)).astype('uint8')
            return Image.fromarray(image)
    raise ValueError('Input image must be a PIL.Image or numpy.ndarray with shape HxW or HxWx3.')


def _make_transform(image_size: Tuple[int, int]):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(opt: object, device: Union[str, torch.device] = None) -> torch.nn.Module:
    model = make_model(opt)
    if device is not None:
        model = model.to(device)
    return model


def build_model_from_config(config_path: str, device: Union[str, torch.device] = None):
    config_path = os.path.abspath(config_path)
    cfg = Config.fromfile(config_path)
    model = build_model(cfg, device=device)
    return model, cfg


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, map_location: Union[str, torch.device] = 'cpu'):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint, strict=False)
    return model


class FPIInference:
    def __init__(self,
                 config_path: str,
                 checkpoint_path: str,
                 device: Union[str, torch.device] = None):
        self.config_path = pathlib.Path(config_path).resolve()
        self.checkpoint_path = pathlib.Path(checkpoint_path).resolve()
        self.device = torch.device(device if device is not None else 'cpu')

        self.cfg = config(self.config_path)
        self.model = build_model(self.cfg, device=self.device)
        load_checkpoint(self.model, self.checkpoint_path, map_location=self.device)
        self.model.eval()

        self.uav_size = tuple(self.cfg.data_config['UAVhw'])
        self.satellite_size = tuple(self.cfg.data_config['Satellitehw'])
        self.uav_transform = _make_transform(self.uav_size)
        self.satellite_transform = _make_transform(self.satellite_size)
        self.use_softmax = self.cfg.model['loss']['cls_loss'].get('use_softmax', False)

    def preprocess(self,
                   uav_image: Union[np.ndarray, Image.Image],
                   satellite_image: Union[np.ndarray, Image.Image],
                   from_bgr: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        uav_pil = _to_pil(uav_image)
        sat_pil = _to_pil(satellite_image)
        if from_bgr:
            uav_pil = Image.fromarray(np.asarray(uav_pil)[:, :, ::-1])
            sat_pil = Image.fromarray(np.asarray(sat_pil)[:, :, ::-1])

        uav_tensor = self.uav_transform(uav_pil).unsqueeze(0)
        sat_tensor = self.satellite_transform(sat_pil).unsqueeze(0)
        return uav_tensor.to(self.device), sat_tensor.to(self.device)

    def _normalize_response(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim == 4 and response.shape[1] > 1:
            if self.use_softmax:
                response = torch.softmax(response, dim=1)[:, 1:]
            else:
                response = torch.sigmoid(response)
        else:
            response = torch.sigmoid(response)
        return response

    def _get_heatmap(self,
                     response: torch.Tensor,
                     output_size: Tuple[int, int] = None) -> np.ndarray:
        response = response.detach().cpu()
        if response.ndim == 4:
            response = response.squeeze(0)
        if response.ndim == 3 and response.shape[0] == 1:
            response = response[0]
        heatmap = response.numpy()
        if output_size is not None and heatmap.shape != output_size:
            import cv2
            heatmap = cv2.resize(heatmap, (output_size[1], output_size[0]))
        return heatmap.astype(np.float32)

    def _argmax_coordinate(self, heatmap: np.ndarray) -> Tuple[int, int]:
        flat_index = np.argmax(heatmap)
        y, x = np.unravel_index(flat_index, heatmap.shape)
        return int(x), int(y)

    def predict(self,
                uav_image: Union[np.ndarray, Image.Image],
                satellite_image: Union[np.ndarray, Image.Image],
                from_bgr: bool = False,
                return_heatmap: bool = False,
                apply_loc_bias: bool = False) -> Dict[str, Union[Tuple[int, int], np.ndarray, Tuple[float, float]]]:
        z, x = self.preprocess(uav_image, satellite_image, from_bgr=from_bgr)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(z, x)

        loc_bias = None
        if isinstance(model_output, tuple) and len(model_output) == 2:
            response, loc_bias = model_output
        else:
            response = model_output

        response = self._normalize_response(response)
        heatmap = self._get_heatmap(response, output_size=self.satellite_size)
        pixel_x, pixel_y = self._argmax_coordinate(heatmap)

        result = {
            'pixel_coordinate': (pixel_x, pixel_y),
            'pixel_normalized': (pixel_x / self.satellite_size[1], pixel_y / self.satellite_size[0]),
            'heatmap': heatmap if return_heatmap else None,
        }

        if apply_loc_bias and loc_bias is not None:
            loc = loc_bias.detach().cpu().squeeze(0).numpy()
            if loc.ndim == 3 and loc.shape[0] == 2:
                orig_heatmap = self._get_heatmap(response, output_size=None)
                base_y, base_x = np.unravel_index(np.argmax(orig_heatmap), orig_heatmap.shape)
                bias_y = float(loc[0, base_y, base_x])
                bias_x = float(loc[1, base_y, base_x])
                corrected_x = (base_x + bias_x) * (self.satellite_size[1] / orig_heatmap.shape[1])
                corrected_y = (base_y + bias_y) * (self.satellite_size[0] / orig_heatmap.shape[0])
                corrected_x = int(np.clip(round(corrected_x), 0, self.satellite_size[1] - 1))
                corrected_y = int(np.clip(round(corrected_y), 0, self.satellite_size[0] - 1))
                result['loc_bias_coordinate'] = (corrected_x, corrected_y)
                result['loc_bias_normalized'] = (corrected_x / self.satellite_size[1], corrected_y / self.satellite_size[0])

        return result


__all__ = [
    'FPIInference',
    'build_model',
    'build_model_from_config',
    'load_checkpoint',
]
