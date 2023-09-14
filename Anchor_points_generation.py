import argparse
import os

import faiss
import numpy as np
from utils import load_pickle, save_pickle, get_data_root


def Anchor_points_generation(args):
    if args.model_type == 'DELG':
        feature_path = os.path.join(get_data_root(), 'train_features/R1M_R101_DELG.pkl')
    elif args.model_type == 'GeM':
        feature_path = os.path.join(get_data_root(), 'train_features/R1M_R101_GeM.pkl')
    else:
        print('>> wrong!')

    train_vecs = np.ascontiguousarray(load_pickle(feature_path)).astype(np.float32)

    print('>> data loaded and start PQ clustering.')
    PQ = faiss.IndexPQ(2048, args.m, args.n_bits)
    PQ.pq.cp.niter = 10
    PQ.pq.cp.nredo = 2
    PQ.pq.cp.max_points_per_centroid = 1000
    PQ.pq.cp.verbose = False
    PQ.train(train_vecs)
    centroids = faiss.vector_to_array(PQ.pq.centroids).reshape(PQ.pq.M, PQ.pq.ksub, PQ.pq.dsub)
    if args.model_type == 'DELG':
        save_path = os.path.join(get_data_root(), 'PQ_centroids/R1M_DELG-R101-Paris-M-PQ_{}_{}_centroids.pkl'.format(args.m, 2**args.n_bits))
    elif args.model_type == 'GeM':
        save_path = os.path.join(get_data_root(), 'PQ_centroids/R1M_GeM-R101-PQ_{}_{}_centroids.pkl'.format(args.m, 2**args.n_bits))
    save_pickle(save_path, centroids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracting R101-DELG Features')
    parser.add_argument('--model_type', type=str, default='DELG')
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--n_bits', type=int, default=8)

    args = parser.parse_args()
    Anchor_points_generation(args)
