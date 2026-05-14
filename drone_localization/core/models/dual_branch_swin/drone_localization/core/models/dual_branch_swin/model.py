import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .loss import CircleLoss

class Gem_heat(nn.Module):
    def __init__(self, dim=49, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 1)
        p = F.softmax(self.p, dim=0).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x

class DualBranchSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_drone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, global_pool='')
        self.branch_sat = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, global_pool='')

        for i in range(2, len(self.branch_drone.layers)):
            self.branch_drone.layers[i] = self.branch_sat.layers[i]

        self.branch_drone.norm = self.branch_sat.norm
        self.num_features = self.branch_drone.num_features
        self.gem_pool = Gem_heat(dim=49)

    def forward(self, x, is_drone):
        feat_seq = torch.zeros(x.shape[0], 49, self.num_features, device=x.device)
        drone_mask = is_drone.bool()
        sat_mask = ~drone_mask

        if drone_mask.any():
            feat_seq[drone_mask] = self.branch_drone(x[drone_mask]).view(-1, 49, self.num_features)

        if sat_mask.any():
            feat_seq[sat_mask] = self.branch_sat(x[sat_mask]).view(-1, 49, self.num_features)

        feat = self.gem_pool(feat_seq)

        return F.normalize(feat, p=2, dim=1)
