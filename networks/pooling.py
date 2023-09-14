import math
import torch.nn as nn

import torch
import torch.nn.functional as F

# --------------------------------------
# pooling
# --------------------------------------


def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))


def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)

    b = (max(H, W) - w) / (steps - 1)
    (_, idx) = torch.min(torch.abs(((w**2 - w * b) / w**2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


class MEAN(nn.Module):
    def __init__(self):
        super(MEAN, self).__init__()

    def forward(self, x, weight=None):
        if weight is not None:
            x *= weight
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))


class MAC(nn.Module):
    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return mac(x)


class RMAC(nn.Module):
    def __init__(self):
        super(RMAC, self).__init__()

    def forward(self, x):
        return rmac(x)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


class Attenpooling(nn.Module):
    def __init__(self, input_dim=512):
        super(Attenpooling, self).__init__()
        self.query = nn.Conv2d(in_channels=input_dim, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        nn.init.xavier_normal_(self.query.weight)
        self.act = nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        attention_map = self.act(self.query(x.detach()))
        x = x * attention_map
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))


class RefineAttenpooling(nn.Module):
    def __init__(self, input_dim=512, alpha=2.0):
        super(RefineAttenpooling, self).__init__()
        self.query = nn.Conv2d(in_channels=input_dim, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        nn.init.xavier_normal_(self.query.weight)
        self.act = nn.Softplus(beta=1, threshold=20)
        self.alpha = alpha

    def forward(self, x):
        b, _, h, w = x.size()
        attention_map = self.act(self.query(x.detach()))
        flatten_x = F.normalize(x, dim=1).flatten(2)
        feature_sim = torch.bmm(flatten_x.permute(0, 2, 1), flatten_x)
        feature_score = F.relu(feature_sim).pow(self.alpha)
        refine_score = torch.bmm(feature_score, attention_map.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view(b, 1, h, w)
        x = x * refine_score
        return F.avg_pool2d(x, (h, w))
