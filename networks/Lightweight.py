import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpfunc import get_data_root, load_pickle
from .pooling import GeM
from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import autocast
import torchvision
import timm
import os

eps_fea_norm = 1e-5
eps_l2_norm = 1e-10


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, eps=1e-6):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.threshold = math.pi - self.m

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, dim=-1), F.normalize(self.weight, dim=-1))
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros(cos_theta.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        selected = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot).bool()

        output = torch.cos(torch.where(selected, theta + self.m, theta))
        output *= self.s
        return output


class efficientnet_b3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        net_in = timm.create_model('tf_efficientnet_b3', num_classes=1000, in_chans=3, pretrained=True, checkpoint_path='')
        features = list(net_in.children())[:-2]
        features.append(nn.Conv2d(1536, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x


class efficientnet_b1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        net_in = timm.create_model('tf_efficientnet_b1', num_classes=1000, in_chans=3, pretrained=True, checkpoint_path='')
        features = list(net_in.children())[:-2]
        features.append(nn.Conv2d(1280, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x


class efficientnet_b0(nn.Module):
    def __init__(self, dim):
        super().__init__()
        net_in = timm.create_model('tf_efficientnet_b0', num_classes=1000, in_chans=3, pretrained=True, checkpoint_path='')
        features = list(net_in.children())[:-2]
        features.append(nn.Conv2d(1280, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x


class mobilenet_v3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        net_in = timm.create_model('mobilenetv3_large_100', num_classes=1000, in_chans=3, pretrained=True, checkpoint_path='')
        features = list(net_in.children())[:-2]
        features.append(nn.Conv2d(1280, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x


class mobilenet_v2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        net_in = timm.create_model('mobilenetv2_100', num_classes=1000, in_chans=3, pretrained=True, checkpoint_path='')
        features = list(net_in.children())[:-2]
        features.append(nn.Conv2d(1280, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x


class shufflenetv2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        net_in = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        features = list(net_in.children())[:-1]
        features.append(nn.Conv2d(1024, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x


class shufflenetv2_small(nn.Module):
    def __init__(self, dim):
        super().__init__()
        net_in = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        features = list(net_in.children())[:-1]
        features.append(nn.Conv2d(1024, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x


class SSP(nn.Module):
    def __init__(self, backbone, PQ_centroid_path):
        super().__init__()
        self.backbone = backbone
        self.pooling = GeM()
        self.whiten = nn.Linear(self.backbone.outputdim, self.backbone.outputdim)
        self.outputdim = self.backbone.outputdim

        # Set up anchor points
        self.m = 64
        self.n_bits = 8

        if PQ_centroid_path is not None:
            PQ_centroids = torch.from_numpy(load_pickle(PQ_centroid_path)).float()
            self.register_buffer('centroids', torch.empty_like(PQ_centroids))
            self.centrooids.data = F.normalize(PQ_centroids, dim=-1)

    @autocast()
    def forward_test(self, x: Tensor):
        x = self.backbone(x)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        x = F.normalize(x, dim=-1)
        x = self.whiten(x)
        x = F.normalize(x, dim=-1)
        return x

    @autocast()
    def forward(self, x: Tensor, features: Tensor):
        B = features.size(0)
        x = self.backbone(x)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        x = F.normalize(x, dim=-1)
        x = self.whiten(x)
        x = F.normalize(x, dim=-1)

        with torch.no_grad():
            features = F.normalize(features.reshape(B, self.m, -1), dim=-1)  # B x M x C
            soft_code = torch.einsum('bmc, mnc -> bmn', [features, self.centroids])
            soft_code = F.softmax(soft_code * 10, dim=-1)

        x = F.normalize(x.reshape(B, self.m, -1), dim=-1)
        x_distance = torch.einsum('bmc, mnc -> bmn', [x, self.centroids.detach()])

        distill = F.kl_div(F.log_softmax(x_distance * 10, dim=-1), soft_code, reduction='none')  # B x m x n
        distill = distill.sum(dim=-1).mean()
        return distill


class HVS(nn.Module):
    def __init__(self, backbone: efficientnet_b3, reduction_dim=2048, classifier_num=1024):
        super().__init__()

        self.backbone = backbone
        self.pooling = GeM()
        self.whiten = nn.Linear(backbone.outputdim, reduction_dim, bias=True)

        self.outputdim = reduction_dim
        self.classifier = ArcFace(in_features=reduction_dim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.15)
        self.oldclassifier = ArcFace(in_features=2048, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.15)

        for p in self.oldclassifier.parameters():
            p.requires_grad_(False)

        state_dict_path = os.path.join(get_data_root(), 'oldclassifier.pkl')
        state_dict = load_pickle(state_dict_path)
        self.oldclassifier.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def forward_test(self, x):
        x = self.backbone(x)
        global_feature = F.normalize(self.pooling(x).squeeze(-1).squeeze(-1), p=2.0, dim=-1)
        global_feature = self.whiten(global_feature)
        global_feature = F.normalize(global_feature, p=2.0, dim=-1)
        return global_feature

    @autocast()
    def forward(self, x, label):
        x = self.backbone(x)
        global_feature = F.normalize(self.pooling(x).squeeze(-1).squeeze(-1), p=2.0, dim=-1)

        global_feature = self.whiten(global_feature)
        global_logits = self.classifier(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)

        global_logits_old = self.oldclassifier(global_feature, label)
        global_loss_old = F.cross_entropy(global_logits_old, label)

        return global_loss, global_logits, global_loss_old, global_logits_old


class RBTBlock(nn.Module):
    """
    The RBT block introduced in <Unified Representation Learning for Cross Model Compatibility>
    """
    def __init__(self, in_planes, out_planes):
        super(RBTBlock, self).__init__()
        self.trans = nn.Sequential(nn.Linear(in_planes, in_planes // 2, bias=False), nn.BatchNorm1d(in_planes // 2, eps=2e-05, momentum=0.9), nn.PReLU(in_planes // 2), nn.Linear(in_planes // 2, out_planes, bias=False),
                                   nn.BatchNorm1d(out_planes, eps=2e-05, momentum=0.9))

    def forward(self, feat):
        output = feat + self.trans(feat)
        return output


class LCE(nn.Module):
    def __init__(self, backbone: efficientnet_b3, reduction_dim=2048, classifier_num=1024):
        super().__init__()

        self.backbone = backbone
        self.pooling = GeM()
        self.whiten = nn.Linear(backbone.outputdim, reduction_dim, bias=True)

        self.outputdim = reduction_dim
        self.classifier = ArcFace(in_features=reduction_dim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.15)
        self.oldclassifier = ArcFace(in_features=2048, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)
        self.trans = RBTBlock(in_planes=reduction_dim, out_planes=reduction_dim)

        for p in self.oldclassifier.parameters():
            p.requires_grad_(False)

        state_dict_path = os.path.join(get_data_root(), 'oldclassifier.pkl')
        state_dict = load_pickle(state_dict_path)
        self.oldclassifier.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def forward_test(self, x):
        x = self.backbone(x)
        global_feature = F.normalize(self.pooling(x).squeeze(-1).squeeze(-1), p=2.0, dim=-1)
        global_feature = self.whiten(global_feature)
        global_feature = self.trans(global_feature)
        global_feature = F.normalize(global_feature, p=2.0, dim=-1)
        return global_feature

    @autocast()
    def forward(self, x, label):
        x = self.backbone(x)
        global_feature = F.normalize(self.pooling(x).squeeze(-1).squeeze(-1), p=2.0, dim=-1)
        global_feature = self.whiten(global_feature)
        global_feature_trans = self.trans(global_feature)
        global_weight_trans = self.trans(self.classifier.weight)
        weight_consistance = torch.sum(torch.pow(F.normalize(global_weight_trans, dim=-1) - F.normalize(self.oldclassifier.weight, dim=-1), 2), dim=-1).clamp(min=1e-6).sqrt().mean()

        global_logits = self.classifier(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)

        global_logits_old = self.oldclassifier(global_feature_trans, label)
        global_loss_old = F.cross_entropy(global_logits_old, label)

        return global_loss, global_logits, global_loss_old, global_logits_old, weight_consistance


class AML(nn.Module):
    def __init__(self, backbone: efficientnet_b3):
        super().__init__()
        self.backbone = backbone
        self.pooling = GeM()
        self.whiten = nn.Linear(self.backbone.outputdim, self.backbone.outputdim)
        self.outputdim = self.backbone.outputdim

    @autocast()
    def forward_test(self, x: Tensor):
        x = self.backbone(x)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        x = F.normalize(x, dim=-1)
        x = self.whiten(x)
        x = F.normalize(x, dim=-1)
        return x

    @autocast()
    def forward(self, x: Tensor, features: Tensor, margin: float):
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x = self.backbone(x)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        x = F.normalize(x, dim=-1)
        x = self.whiten(x)
        x = F.normalize(x, dim=-1)
        x = x.view(B, N, -1)

        # Contrastive loss
        anchor = x[:, 0, :].unsqueeze(1)
        positive_t = features[:, 1, :].unsqueeze(1)
        negtative_t = features[:, 1:, :]
        dis_ap_t = torch.sum(torch.pow(anchor - positive_t, 2), dim=-1).clamp(min=1e-6).sqrt()
        dis_an_t = torch.sum(torch.pow(anchor - negtative_t, 2), dim=-1).clamp(min=1e-6).sqrt()
        contrast_t = torch.sum(0.5 * torch.pow(dis_ap_t, 2) + 0.5 * torch.pow(margin - dis_an_t, 2), dim=-1).mean()

        # Regression loss
        reg = torch.sum(torch.pow(x.view(B * N, -1) - features.view(B * N, -1), 2), dim=-1).clamp(min=1e-6).sqrt().mean()
        return contrast_t, reg
