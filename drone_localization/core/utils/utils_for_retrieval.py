import os

import torch
import yaml

from ..models.ConvNext import two_view_net


def get_model_list(dirname, key):
    """Получение последнего файла модели"""
    if not os.path.exists(dirname):
        print("No dir: %s" % dirname)
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


def load_network(name, opt, dirname="./data/convnext_tri/weights/"):
    """Загрузка обученной сети"""

    # Находим последнюю сохраненную модель
    last_model_name = os.path.basename(get_model_list(dirname, "net"))
    if last_model_name is None:
        raise FileNotFoundError(f"No .pth files found in {dirname}")

    epoch = last_model_name.split("_")[1].split(".")[0]
    if epoch != "last":
        epoch = int(epoch)

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
    if isinstance(epoch, int):
        save_filename = "net_%03d.pth" % epoch
    else:
        save_filename = "net_%s.pth" % epoch

    save_path = os.path.join(dirname, save_filename)
    print("Loading model from %s" % save_path)

    state_dict = torch.load(save_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    return model, opt, epoch
