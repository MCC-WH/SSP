import torch.nn as nn
import torchvision
from torch import nn
from .pooling import GeM
import torch
import torch.nn.functional as F


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        net_in = getattr(torchvision.models, 'resnet101')(pretrained=False)
        features = list(net_in.children())[:-2]
        features = nn.Sequential(*features)
        self.outputdim = 2048
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        self.block5 = features[7]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        net_in = getattr(torchvision.models, 'resnet50')(pretrained=False)
        features = list(net_in.children())[:-2]
        features = nn.Sequential(*features)
        self.outputdim = 2048
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        self.block5 = features[7]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class DELG(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet101()
        self.pooling = GeM()
        self.whiten = nn.Conv2d(self.backbone.outputdim, self.backbone.outputdim, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.outputdim = self.backbone.outputdim

    def forward_test(self, x):
        x = self.backbone(x)
        global_feature = F.normalize(self.pooling(x), p=2.0, dim=1)
        global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        global_feature = F.normalize(global_feature, p=2.0, dim=-1)
        return global_feature

    def forward(self, x):
        x = self.backbone(x)
        global_feature = F.normalize(self.pooling(x), p=2.0, dim=1)
        global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        global_feature = F.normalize(global_feature, p=2.0, dim=-1)
        return global_feature