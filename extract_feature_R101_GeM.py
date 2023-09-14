import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url
from torchvision import transforms
from tqdm import tqdm

from Dataset import ImageFromList, RoxfordAndRparis, cid2filename
from networks import init_network
from utils import (compute_map_and_print, get_data_root, load_pickle,
                   save_pickle)

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}


def whitenapply(X, m, P, dimensions=None):

    if not dimensions:
        dimensions = P.shape[0]

    X = np.dot(P[:dimensions, :], X - m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)

    return X


@torch.no_grad()
def test(data_root, net, datasets=['roxford5k'], device=torch.device('cuda'), ms=[1], msp=1.0, Lw=None):
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
        feature_prefix = os.path.join(dataset_dir, 'R101-GeM-{}.pkl'.format(dataset))
        query_loader = DataLoader(ImageFromList(Image_paths=qimages, transforms=transform, imsize=image_size, bbox=bbxs), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        db_loader = DataLoader(ImageFromList(Image_paths=images, transforms=transform, imsize=image_size, bbox=None), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        # extract database and query vectors
        vecs = extract_vectors(net=net, loader=db_loader, device=device, ms=ms, msp=msp)
        qvecs = extract_vectors(net=net, loader=query_loader, device=device, ms=ms, msp=msp)

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        if Lw is not None:
            # whiten the vectors
            vecs = whitenapply(vecs.T, Lw['m'], Lw['P'])
            vecs = vecs.T
            qvecs = whitenapply(qvecs.T, Lw['m'], Lw['P'])
            qvecs = qvecs.T

        save_pickle(feature_prefix, {'db': vecs, 'query': qvecs})

        # search, rank, and print
        scores = np.dot(vecs, qvecs.T)
        ranks = np.argsort(-scores, axis=0)
        mapE, mapM, mapH = compute_map_and_print(dataset, 'R101-GeM', 'whitening', ranks, cfg['gnd'])


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


def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist


def ExtractFeature(args):
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(">> Loading network:\n>>>> '{}'".format(args.network))
    print(">> data root:{}".format(get_data_root()))
    state = load_url(PRETRAINED[args.network], model_dir=os.path.join(get_data_root(), 'networks'))
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']  # 'mac' 'spoc' 'gem' 'rmac'
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False

    # network initialization
    net = init_network(net_params)
    net.load_state_dict(state['state_dict'], strict=True)

    if 'Lw' in state['meta']:
        net.meta['Lw'] = state['meta']['Lw']

    ms = list(eval(args.multiscale))
    if len(ms) > 1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
    else:
        msp = 1

    # moving network to gpu and eval mode
    net.to(device)
    net.eval()

    # compute whitening
    if 'Lw' in net.meta:
        print('>> {}: Whitening is precomputed, loading it...'.format(args.network))

        if len(ms) > 1:
            Lw = net.meta['Lw']['retrieval-SfM-120k']['ms']
        else:
            Lw = net.meta['Lw']['retrieval-SfM-120k']['ss']

    else:
        Lw = None

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

    test(data_root=get_data_root(), net=net, datasets=['roxford5k', 'rparis6k'], device=device, ms=ms, msp=msp, Lw=Lw)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_loader = DataLoader(ImageFromList(Image_paths=train_images, imsize=1024, transforms=transform), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(ImageFromList(Image_paths=val_images, imsize=1024, transforms=transform), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    train_vecs = extract_vectors(net=net, loader=train_loader, device=device, ms=ms, msp=msp)
    train_vecs = train_vecs.numpy()
    if Lw is not None:
        train_vecs = whitenapply(train_vecs.T, Lw['m'], Lw['P'])
        train_vecs = train_vecs.T

    val_vecs = extract_vectors(net=net, loader=val_loader, device=device, ms=ms, msp=msp)
    val_vecs = val_vecs.numpy()
    if Lw is not None:
        val_vecs = whitenapply(val_vecs.T, Lw['m'], Lw['P'])
        val_vecs = val_vecs.T
    if args.dataset == 'retrieval-SfM-120k':
        feature_prefix = os.path.join(get_data_root(), 'train_features/SFM_R101_GeM.pkl')
    elif args.dataset == 'GLDv2':
        feature_prefix = os.path.join(get_data_root(), 'train_features/GLDv2_R101_GeM.pkl')
    else:
        raise ValueError('Unsupport dataset type')
    save_pickle(feature_prefix, {'train': train_vecs, 'val': val_vecs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracting R101-GeM Features')
    # test options
    parser.add_argument('--dataset', type=str, default='retrieval-SfM-120k')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--network', default='retrievalSfM120k-resnet101-gem', metavar='NETWORK', help="network to be evaluated: " + " | ".join(PRETRAINED.keys()))
    parser.add_argument('--image_size', default=1024, type=int, metavar='N', help="maximum size of longer image side used for testing (default: 1024)")
    parser.add_argument('--multiscale', type=str, metavar='MULTISCALE', default='[1]', help="use multiscale vectors for testing, " + " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")

    args = parser.parse_args()
    ExtractFeature(args)