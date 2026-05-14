import torch
import torch.nn as nn

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, feat, labels):
        sim = torch.matmul(feat, feat.T)
        lbl = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_m = lbl & ~torch.eye(labels.size(0), dtype=torch.bool, device=feat.device)
        neg_m = ~lbl
        sp, sn = sim[pos_m], sim[neg_m]
        if len(sp) == 0 or len(sn) == 0: return (feat * 0.0).sum()

        ap = torch.relu(-sp.detach() + 1 + self.m)
        an = torch.relu(sn.detach() + self.m)

        lp = -self.gamma * ap * (sp - (1 - self.m))
        ln = self.gamma * an * (sn - self.m)
        return torch.logsumexp(ln, dim=0) + torch.logsumexp(lp, dim=0)
