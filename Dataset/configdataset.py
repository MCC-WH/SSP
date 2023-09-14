import os
import pickle

DATASETS = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']


def RoxfordAndRparis(dataset, dir_main):
    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    # loading imlist, qimlist, and gnd, in cfg as a dict
    gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
    cfg['gnd_fname'] = gnd_fname

    cfg['ext'] = '.jpg'
    cfg['qext'] = '.jpg'
    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = []
    for name in cfg['imlist']:
        cfg['im_fname'].append(os.path.join(cfg['dir_images'], name + '.jpg'))

    cfg['qim_fname'] = []
    for name in cfg['qimlist']:
        cfg['qim_fname'].append(os.path.join(cfg['dir_images'], name + '.jpg'))

    cfg['dataset'] = dataset

    return cfg
