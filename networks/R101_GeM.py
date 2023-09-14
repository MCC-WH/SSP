import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(2), x.size(3))).pow(1.0 / p)


def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


class GEM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GEM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


class L2N(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return l2n(x, self.eps)


class R101_GeM(nn.Module):
    def __init__(self, features, lwhiten=None, pool=None, whiten=None, meta=None):
        super().__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta

    def forward(self, x):
        o = self.features(x)

        if self.lwhiten is not None:
            s = o.size()
            o = o.permute(0, 2, 3, 1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0], s[2], s[3], self.lwhiten.out_features).permute(0, 3, 1, 2)

        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        if self.whiten is not None:
            o = self.norm(self.whiten(o))
        return o


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])

    # get output dimensionality size
    dim = 2048
    # initialize with random weights
    net_in = getattr(torchvision.models, architecture)(pretrained=True)
    features = list(net_in.children())[:-2]
    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
    else:
        lwhiten = None

    # initialize pooling
    pool = GEM()

    # initialize whitening
    if whitening:
        whiten = nn.Linear(dim, dim, bias=True)
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture': architecture,
        'local_whitening': local_whitening,
        'pooling': pooling,
        'regional': regional,
        'whitening': whitening,
        'mean': mean,
        'std': std,
        'outputdim': dim,
    }

    # create a generic image retrieval network
    net = R101_GeM(features, lwhiten, pool, whiten, meta)
    return net
