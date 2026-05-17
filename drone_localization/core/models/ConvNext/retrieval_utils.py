import logging
import os

import torch
import yaml

from .model import two_view_net  # локальный импорт

logger = logging.getLogger(__name__)


def get_model_list(dirname, key):
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


def load_network(name, opt, weights_dir, model_weights="net_152.pth"):
    # weights_dir уже передан, не вычисляем заново
    last_model_name = "net_152.pth"
    config_path = os.path.join(weights_dir, "opts.yaml")
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    for key, value in config.items():
        if hasattr(opt, key):
            setattr(opt, key, value)
    if opt.views == 2:
        model = two_view_net(
            opt.nclasses, block=opt.block, return_f=False, resnet=opt.resnet
        )
    else:
        raise ValueError(f"Unknown views: {opt.views}")
    save_path = os.path.join(weights_dir, model_weights)
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model not found: {save_path}")
    logger.info("Loading model from %s" % save_path)
    state_dict = torch.load(save_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return model, opt, None