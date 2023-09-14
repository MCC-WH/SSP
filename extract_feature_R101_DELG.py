import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Dataset import ImageFromList, RoxfordAndRparis, cid2filename
from networks.R101_DELG import DELG
from utils import (compute_map_and_print, get_data_root, load_pickle, save_pickle)


@torch.no_grad()
def test(data_root, net, datasets=['roxford5k'], device=torch.device('cuda'), ms=[1], msp=1.0):
    image_size = 1024
    net.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # evaluate on test datasets
    for dataset in datasets:
        # prepare config structure for the test dataset
        cfg = RoxfordAndRparis(dataset, os.path.join(data_root, "test"))
        images = cfg['im_fname']
        qimages = cfg['qim_fname']
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

        dataset_dir = os.path.join(get_data_root(), 'test_features')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        feature_prefix = os.path.join(dataset_dir, 'R101-DELG-{}.pkl'.format(dataset))
        query_loader = DataLoader(ImageFromList(Image_paths=qimages, transforms=transform, imsize=image_size, bbox=bbxs), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        db_loader = DataLoader(ImageFromList(Image_paths=images, transforms=transform, imsize=image_size, bbox=None), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        # extract database and query vectors
        vecs = extract_vectors(net=net, loader=db_loader, device=device, ms=ms, msp=msp)
        qvecs = extract_vectors(net=net, loader=query_loader, device=device, ms=ms, msp=msp)

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        save_pickle(feature_prefix, {'db': vecs, 'query': qvecs})

        # search, rank, and print
        scores = np.dot(vecs, qvecs.T)
        ranks = np.argsort(-scores, axis=0)
        mapE, mapM, mapH = compute_map_and_print(dataset, 'R101-DELG', 'whitening', ranks, cfg['gnd'])


@torch.no_grad()
def extract_vectors(net, loader, device, ms=[1], msp=1):
    vecs = torch.zeros(len(loader), net.meta['outputdim'])
    for i, input in tqdm(enumerate(loader), total=len(loader)):
        input = input.to(device)

        if len(ms) == 1 and ms[0] == 1:
            vecs[i, :] = net(input).cpu().data.squeeze()
        else:
            v = torch.zeros(net.meta['outputdim'])
            for s in ms:
                if s == 1:
                    input_t = input.clone()
                else:
                    input_t = F.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
                v += net(input_t).pow(msp).cpu().data.squeeze()
            v /= len(ms)
            v = v.pow(1. / msp)
            v /= v.norm()
            vecs[i, :] = v
    return vecs


def ExtractFeature(args):
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(">> data root:{}".format(get_data_root()))

    # network initialization
    net = DELG()
    net.load_state_dict(os.path.join(get_data_root(), 'R101-DELG.pth'), strict=True)

    ms = list(eval(args.multiscale))
    msp = 1

    # moving network to gpu and eval mode
    net.to(device)
    net.eval()

    if args.dataset == 'retrieval-SfM-120k':
        ims_root = os.path.join(get_data_root(), "/train/retrieval-SfM-120k/ims/")
        db_fn = os.path.join(get_data_root(), "/train/retrieval-SfM-120k/retrieval-SfM-120k.pkl")
        db = load_pickle(db_fn)
        train_images = [cid2filename(db['train']['cids'][i], ims_root) for i in range(len(db['train']['cids']))]
        val_images = [cid2filename(db['val']['cids'][i], ims_root) for i in range(len(db['val']['cids']))]
    elif args.dataset == 'GLDv2':
        prefix_train = os.path.join(get_data_root(), 'train', 'GLDv2', 'GLDv2-clean-train-split.pkl')
        prefix_val = os.path.join(get_data_root(), 'train', 'GLDv2', 'GLDv2-clean-val-split.pkl')
        train_images = load_pickle(prefix_train)['image_paths']
        val_images = load_pickle(prefix_val)['image_paths']
    else:
        raise ValueError('Unsupport training dataset')

    test(data_root=get_data_root(), net=net, datasets=['roxford5k', 'rparis6k'], device=device, ms=ms, msp=msp)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_loader = DataLoader(ImageFromList(Image_paths=train_images, imsize=1024, transforms=transform), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(ImageFromList(Image_paths=val_images, imsize=1024, transforms=transform), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    train_vecs = extract_vectors(net=net, loader=train_loader, device=device, ms=ms, msp=msp)
    train_vecs = train_vecs.numpy()
    val_vecs = extract_vectors(net=net, loader=val_loader, device=device, ms=ms, msp=msp)
    val_vecs = val_vecs.numpy()
    if args.dataset == 'retrieval-SfM-120k':
        feature_prefix = os.path.join(get_data_root(), 'train_features/SFM_R101_DELG.pkl')
    elif args.dataset == 'GLDv2':
        feature_prefix = os.path.join(get_data_root(), 'train_features/GLDv2_R101_DELG.pkl')
    else:
        raise ValueError('Unsupport dataset type')
    save_pickle(feature_prefix, {'train': train_vecs, 'val': val_vecs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracting R101-DELG Features')
    # test options
    parser.add_argument('--dataset', type=str, default='retrieval-SfM-120k')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', default=1024, type=int, metavar='N', help="maximum size of longer image side used for testing (default: 1024)")
    parser.add_argument('--multiscale', type=str, metavar='MULTISCALE', default='[1]', help="use multiscale vectors for testing, " + " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")

    args = parser.parse_args()
    ExtractFeature(args)