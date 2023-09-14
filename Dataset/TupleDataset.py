import os
import time
from utils import get_data_root, load_pickle
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .ImageFromList import GLDv2_build_contrastive_dataset
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def imcrop(img, params):
    img = transforms.functional.crop(img, *params)
    return img


def imthumbnail(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img


def imresize(img, imsize):
    img = transforms.Resize(imsize)(img)
    return img


def cid2filename(cid, prefix):
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


class FeatureFromList(data.Dataset):
    def __init__(self, features=None, topk_indices=None):
        super(FeatureFromList, self).__init__()
        self.features = features
        self.topk_indices = topk_indices
        self.len = topk_indices.size(0)

    def __getitem__(self, index):
        topk_indices = self.topk_indices[index]
        feature = self.features[topk_indices]
        return feature

    def __len__(self):
        return self.len


class ImageFromList(data.Dataset):
    def __init__(self, Image_paths=None, transforms=None, imsize=None, bbox=None, loader=pil_loader):
        super(ImageFromList, self).__init__()
        self.Image_paths = Image_paths
        self.transforms = transforms
        self.bbox = bbox
        self.imsize = imsize
        self.loader = loader
        self.len = len(Image_paths)

    def __getitem__(self, index):
        path = self.Image_paths[index]
        img = self.loader(path)
        imfullsize = max(img.size)

        if self.bbox is not None:
            img = img.crop(self.bbox[index])

        if self.imsize is not None:
            if self.bbox is not None:
                img = imthumbnail(img, self.imsize * max(img.size) / imfullsize)
            else:
                img = imthumbnail(img, self.imsize)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return self.len


class Tuples_Distill(data.Dataset):
    def __init__(self, name, mode, imsize=None, qsize=2000, nnum=5, poolsize=20000, feature_path=None):

        if name == 'retrieval-SfM-120k':
            # SFM images
            ims_root = os.path.join(get_data_root(), "/train/retrieval-SfM-120k/ims/")
            db_fn = os.path.join(get_data_root(), "/train/retrieval-SfM-120k/retrieval-SfM-120k.pkl")
            db = load_pickle(db_fn)[mode]
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
            self.total_length = len(self.images)

            # SFM features
            self.features = torch.tensor(load_pickle(feature_path)[mode]).float()

            self.transform = transforms.Compose([transforms.Resize(size=imsize), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif name == "GLDv2":
            db_fn = os.path.join(get_data_root(), 'train', 'GLDv2', 'GLDv2_Triplet.pkl')
            if not os.path.exists(db_fn):
                GLDv2_build_contrastive_dataset()
            db = load_pickle(db_fn)[mode]
            self.images = db['cids']

            # GLDv2 features
            self.features = torch.tensor(load_pickle(feature_path)[mode]).float()
            self.total_length = len(self.images)

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=imsize, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # initializing tuples dataset
        self.imsize = imsize
        self.clusters = db['cluster']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']

        # size of training subset for an epoch
        self.nnum = nnum
        self.qsize = min(qsize, len(self.images))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None
        self.eval_transform = transforms.Compose([transforms.Resize(size=(imsize, imsize)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.loader = pil_loader

        self.print_freq = 10

    def __getitem__(self, index):
        images = []
        features = []

        # query image
        images.append(self.transform(self.loader(self.images[self.qidxs[index]])))
        features.append(self.features[self.qidxs[index]])
        # positive features
        images.append(self.transform(self.loader(self.images[self.pidxs[index]])))
        features.append(self.features[self.pidxs[index]])
        # negative features
        for i in range(len(self.nidxs[index])):
            images.append(self.transform(self.loader(self.images[self.nidxs[index][i]])))
            features.append(self.features[self.nidxs[index][i]])
        images = torch.stack(images)
        features = torch.stack(features)
        return images, features

    def __len__(self):
        return self.qsize

    @torch.no_grad()
    def create_epoch_tuples(self, net, device=torch.device('cuda')):

        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = [self.ppool[i] for i in idxs2qpool]

        idxs2images = torch.randperm(len(self.images))[:self.poolsize]
        net.to(device)
        net.eval()

        loader = DataLoader(ImageFromList(Image_paths=[self.images[i] for i in self.qidxs], imsize=self.imsize, transforms=self.eval_transform), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        qvecs = torch.zeros(net.outputdim, self.qsize)
        for i, input in enumerate(loader):
            qvecs[:, i] = net.forward_test(input.to(device)).data.squeeze().cpu()
            if (i + 1) % self.print_freq == 0 or (i + 1) == len(self.qidxs):
                print('\r>>>> {}/{} done...'.format(i + 1, len(self.qidxs)), end='')

        poolvecs = self.features[idxs2images].t()

        scores = torch.mm(poolvecs.t(), qvecs)
        scores, ranks = torch.sort(scores, dim=0, descending=True)
        avg_ndist = torch.tensor(0).float()
        n_ndist = torch.tensor(0).float()
        self.nidxs = []
        for q in range(len(self.qidxs)):
            qcluster = self.clusters[self.qidxs[q]]
            clusters = [qcluster]
            nidxs = []
            r = 0
            while len(nidxs) < self.nnum:
                potential = idxs2images[ranks[r, q]]
                if not self.clusters[potential] in clusters:
                    nidxs.append(potential)
                    clusters.append(self.clusters[potential])
                    avg_ndist += torch.pow(qvecs[:, q] - poolvecs[:, ranks[r, q]] + 1e-6, 2).sum(dim=0).sqrt()
                    n_ndist += 1
                r += 1
            self.nidxs.append(nidxs)
        return (avg_ndist / n_ndist).item()