import os

import torch
import yaml
import logging

from ..models.ConvNext import two_view_net

logger = logging.getLogger(__name__)

def get_model_list(dirname, key):
    """Получение последнего файла модели"""
    if not os.path.exists(dirname):
        logger.info("No dir: %s" % dirname)
        return None

    gen_models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f
    ]

    if not gen_models:
        return None

    gen_models.sort()
    return gen_models[-1]


def load_network(name, opt, dirname):
    """Загрузка обученной сети"""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    dirname = os.path.join(project_root, "weights")
    # Находим сохраненную модель
    last_model_name = "net_152.pth"

    # Загружаем конфигурацию
    config_path = os.path.join(dirname, "opts.yaml")
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    # Применяем конфигурацию к opt (только существующие атрибуты)
    for key, value in config.items():
        if hasattr(opt, key):
            setattr(opt, key, value)

    # Создаем модель
    if opt.views == 2:
        model = two_view_net(
            opt.nclasses, block=opt.block, return_f=False, resnet=opt.resnet
        )
    else:
        raise ValueError(f"Unknown views: {opt.views}")

    # Загружаем веса
    save_path = os.path.join(dirname, last_model_name)
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model not found: {save_path}")
    
    logger.info("Loading model from %s" % save_path)

    state_dict = torch.load(save_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    return model, opt, None 
