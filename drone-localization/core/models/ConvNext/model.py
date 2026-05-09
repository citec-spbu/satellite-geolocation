import torch.nn as nn
from core.models.ConvNext.make_model import make_convnext_model


class two_view_net(nn.Module):
    """Модель для двух видов (satellite + drone)"""

    def __init__(self, class_num, block=4, return_f=False, resnet=False):
        super(two_view_net, self).__init__()
        self.model_1 = make_convnext_model(num_class=class_num, block=block,
                                           return_f=return_f, resnet=resnet)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_1(x2)
        return y1, y2

    def get_embedding(self, x, view='drone'):
        """Извлечение эмбеддинга для конкретного вида"""
        if view == 'drone':
            _, features = self.forward(None, x)
        elif view == 'satellite':
            features, _ = self.forward(x, None)
        else:
            raise ValueError(f"Unknown view: {view}")

        # features - это кортеж ([B,512,block+1], [B,512])
        if isinstance(features, tuple):
            return features[1]  # основной эмбеддинг
        return features