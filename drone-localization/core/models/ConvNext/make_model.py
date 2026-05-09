import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.models.ConvNext.backbone_convnext import convnext_tiny


class Gem_heat(nn.Module):
    """Generalized Mean Pooling with learnable p"""

    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


class BasicConv(nn.Module):
    """Базовый сверточный блок"""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    """Concatenate max and mean pooling"""

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    """Attention gate для Triplet Attention"""

    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    """Triplet Attention механизм"""

    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        return x_out11, x_out21


class ClassBlock(nn.Module):
    """Классификационный блок с возможностью возврата эмбеддинга"""

    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True,
                 num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f

        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        f = self.add_block(x)  # эмбеддинг

        if self.training and not self.return_f:
            return self.classifier(f)
        elif self.training and self.return_f:
            return self.classifier(f), f
        else:
            return f  # на инференсе возвращаем только эмбеддинг


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class build_convnext(nn.Module):
    """Полная модель с ConvNeXt backbone и Triplet Attention"""

    def __init__(self, num_classes, block=4, return_f=False, resnet=False):
        super(build_convnext, self).__init__()
        self.return_f = return_f

        if resnet:
            raise NotImplementedError("ResNet backbone временно не поддерживается")
        else:
            print('Using model_type: convnext_tiny as a backbone')
            self.in_planes = 768
            self.convnext = convnext_tiny(pretrained=True)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block

        if block > 0:
            self.tri_layer = TripletAttention()
            for i in range(self.block):
                name = 'classifier_mcb' + str(i + 1)
                setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

    def forward(self, x):
        # Получаем признаки из ConvNeXt
        gap_feature, part_features = self.convnext.forward_features(x)

        # Основной эмбеддинг через classifier1
        convnext_feature = self.classifier1(gap_feature)  # [B, 512]

        if self.block == 0:
            return convnext_feature

        # Triplet Attention на пространственных признаках
        tri_features = self.tri_layer(part_features)

        # Извлекаем части
        tri_list = []
        for i in range(self.block):
            tri_list.append(tri_features[i].mean([-2, -1]))
        triatten_features = torch.stack(tri_list, dim=2)  # [B, 768, block]

        # Классифицируем каждую часть
        part_features_list = []
        for i in range(self.block):
            part = triatten_features[:, :, i].view(triatten_features.size(0), -1)
            classifier = getattr(self, 'classifier_mcb' + str(i + 1))
            part_features_list.append(classifier(part))  # каждый [B, 512]

        # Объединяем все эмбеддинги
        y = torch.stack(part_features_list, dim=2)  # [B, 512, block]
        ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)  # [B, 512, 1]
        y = torch.cat([y, ffeature], dim=2)  # [B, 512, block+1]

        return y, convnext_feature  # возвращаем и тензор, и основной эмбеддинг

    def get_embedding(self, x):
        """Извлечение только эмбеддинга (для инференса)"""
        gap_feature, _ = self.convnext.forward_features(x)
        return self.classifier1(gap_feature)  # [B, 512]


def make_convnext_model(num_class, block=4, return_f=False, resnet=False):
    """Фабрика моделей"""
    print('===========building convnext===========')
    model = build_convnext(num_class, block=block, return_f=return_f, resnet=resnet)
    return model