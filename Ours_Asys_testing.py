import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks import DELG, SSP
from Dataset import ImageFromList, RoxfordAndRparis
from networks.Lightweight import mobilenet_v2
from utils import (compute_map_and_print, extract_vectors, load_pickle, save_pickle)
from utils.helpfunc import get_checkpoint_root, get_data_root, save_pickle

warnings.filterwarnings('ignore')


def imthumbnail(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img


@torch.no_grad()
def extract_vectors(net, loader, ms=[1], device=torch.device('cuda')):
    net.eval()
    total = len(loader)
    vecs = torch.zeros(total, net.outputdim)
    if len(ms) == 1:
        for i, input in tqdm(enumerate(loader), total=total):
            batch_size_inner = input.shape[0]
            vecs[i * batch_size_inner:((i + 1) * batch_size_inner), :] = net.forward_test(input.to(device)).cpu().data.squeeze()
    else:
        for i, input in tqdm(enumerate(loader), total=total):
            batch_size_inner = input.shape[0]
            vec = torch.zeros(batch_size_inner, net.outputdim)
            for s in ms:
                if s == 1:
                    input_ = input.clone()
                else:
                    input_ = F.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
                vec += net.forward_test(input_.to(device)).cpu().data.squeeze()
            vec /= len(ms)
            vecs[i * batch_size_inner:((i + 1) * batch_size_inner), :] = F.normalize(vec, p=2, dim=-1)
    return vecs


def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageFromList(data.Dataset):
    def __init__(self, Image_paths=None, transforms=None, imsize=None, bbox=None, loader=pil_loader):
        super(ImageFromList, self).__init__()
        self.Image_paths = Image_paths
        self.transforms = transforms
        self.imsize = imsize
        self.loader = loader
        self.len = len(Image_paths)

    def __getitem__(self, index):
        path = self.Image_paths[index]
        img = self.loader(path)
        img = imthumbnail(img, self.imsize)
        img = self.transforms(img)

        return img

    def __len__(self):
        return self.len


@torch.no_grad()
def test(datasets, query_net=None, gallery_net=None, device=torch.device('cuda'), ms=[1, 2**(1 / 2), (1 / 2)**(1 / 2)], pool='GeM asys', whiten='whiten'):
    image_size = 1024
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    query_net.eval()
    gallery_net.eval()
    for dataset in datasets:
        # prepare config structure for the test dataset
        cfg = RoxfordAndRparis(dataset, os.path.join(get_data_root(), 'test'))
        db_images = cfg['im_fname']
        qimages = cfg['qim_fname']
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

        query_loader = DataLoader(ImageFromList(Image_paths=qimages, transforms=transform, imsize=image_size, bbox=bbxs), batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
        qvecs = extract_vectors(query_net, query_loader, ms, device)
        qvecs = qvecs.numpy()

        db_loader = DataLoader(ImageFromList(Image_paths=db_images, transforms=transform, imsize=image_size, bbox=None), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        vecs = extract_vectors(gallery_net, db_loader, ms, device)
        vecs = vecs.numpy()
        scores = np.dot(vecs, qvecs.T)
        ranks = np.argsort(-scores, axis=0)
        _, mapM, mapH = compute_map_and_print(dataset, pool, whiten, ranks, cfg['gnd'])


if __name__ == "__main__":
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    db_net = DELG()
    db_net.load_state_dict(os.path.join(get_data_root(), 'R101-DELG.pth'), strict=True)
    db_net.eval()
    backbone = mobilenet_v2(2048)
    query_net = SSP(backbone=backbone, PQ_centroid_path=None)
    resume = os.path.join(get_checkpoint_root(), 'GLDv2-mobilenet_v2-Ours', 'best_checkpoint.pth')
    checkpoint = torch.load(resume)
    query_net.load_state_dict(checkpoint['state_dict'], strict=False)
    query_net.eval()
    result = test(['roxford5k', 'rparis6k'], query_net=query_net, gallery_net=db_net)
