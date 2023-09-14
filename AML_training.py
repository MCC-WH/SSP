import argparse
import math
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from networks import mobilenet_v2, efficientnet_b3
from torch import cuda, optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from Dataset.TupleDataset import Tuples_Distill
from networks import AML
from utils import (MetricLogger, NumberLogger, create_optimizer, get_checkpoint_root, get_data_root, get_rank, init_distributed_mode, is_main_process)

warnings.filterwarnings('ignore')


def set_deterministic(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cuda.manual_seed(seed)


def collate_tuples(batch):
    batch = list(filter(lambda x: x is not None, batch))
    image, feature = zip(*batch)
    return torch.stack(image), torch.stack(feature, dim=0)


class WarmupCos_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(math.pi * np.arange(decay_iter) / decay_iter))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        return self.lr_schedule[self.iter]

    def state_dict(self):
        state_dict = {}
        state_dict['base_lr'] = self.base_lr
        state_dict['lr_schedule'] = self.lr_schedule
        state_dict['iter'] = self.iter
        return state_dict

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict['base_lr']
        self.lr_schedule = state_dict['lr_schedule']
        self.iter = state_dict['iter']


def main(args):
    init_distributed_mode(args)
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))
    if args.device == 'cuda':
        device = torch.device('cuda' if cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    args.directory = os.path.join(get_checkpoint_root(), '{}-{}-AML'.format(args.dataset, args.network))
    os.makedirs(args.directory, exist_ok=True)

    if args.distributed:
        ngpus_per_node = cuda.device_count()
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print('>> batch size per node:{}'.format(args.batch_size))
        print('>> num workers per node:{}'.format(args.num_workers))

    if args.dataset == 'retrieval-SfM-120k':
        feature_path = os.path.join(get_data_root(), 'train_features/SFM_R101_DELG.pkl')
    elif args.dataset == "GLDv2":
        feature_path = os.path.join(get_data_root(), 'train_features/GLDv2_R101_DELG.pkl')
    else:
        raise ValueError('Unsupport training dataset')

    train_dataset = Tuples_Distill(name=args.dataset, mode='train', imsize=args.imsize, qsize=2000, nnum=5, poolsize=20000, feature_path=feature_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=False, collate_fn=collate_tuples)
    val_dataset = Tuples_Distill(name=args.dataset, mode='val', imsize=args.imsize, qsize=2000, nnum=5, poolsize=20000, feature_path=feature_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=False, collate_fn=collate_tuples)

    if args.network == 'mobilenet_v2':
        backbone = mobilenet_v2(2048)
    elif args.network == 'efficientnet_b3':
        backbone = efficientnet_b3(2048)
    else:
        raise ValueError('Unsupport backbone type')
    model = AML(backbone=backbone).to(device)
    model_without_ddp = model

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('>> number of params:{:.2f}M'.format(n_parameters / (1024 * 1024)))

    # define optimizer
    param_dicts = create_optimizer(args.weight_decay, model_without_ddp)
    optimizer = optim.SGD(param_dicts, lr=args.base_lr * args.batch_size / 256, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    lr_scheduler = WarmupCos_Scheduler(optimizer=optimizer,
                                       warmup_epochs=args.warmup_epochs,
                                       warmup_lr=args.warmup_lr * args.batch_size * args.update_every / 256,
                                       num_epochs=args.num_epochs,
                                       base_lr=args.base_lr * args.batch_size * args.update_every / 256,
                                       final_lr=args.final_lr * args.batch_size * args.update_every / 256,
                                       iter_per_epoch=int(len(train_loader) / args.update_every))
    lr_scheduler.iter = max(int(len(train_loader) / args.update_every) * start_epoch, 0)

    # Start training
    metric_logger = MetricLogger(delimiter=" ")
    val_metric_logger = MetricLogger(delimiter=" ")
    print_freq = 10
    model_path = None
    scaler = GradScaler()
    Loss_logger = NumberLogger()
    Val_Loss_logger = NumberLogger()

    for epoch in range(start_epoch, args.num_epochs):
        set_deterministic(seed=int(epoch + get_rank() + args.seed))
        avg_neg_distance = train_loader.dataset.create_epoch_tuples(model_without_ddp, device)
        Loss_logger.meters['negative distance'].append((epoch * len(train_loader), avg_neg_distance))
        header = '>> Train Epoch: [{}]'.format(epoch)

        optimizer.zero_grad()
        for idx, (images, features) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            model.train()
            images = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            contrast, reg = model(images, features, args.margin)
            loss = contrast + reg

            if not math.isfinite(contrast.item()):
                print(">> Contrast loss is nan, skipping")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            if not math.isfinite(reg.item()):
                print(">> Reg loss is nan, skipping")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            metric_logger.meters['reg loss'].update(reg.item())
            metric_logger.meters['contrast loss'].update(contrast.item())

            if (idx + 1) % args.update_every == 0:
                if args.clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                _ = lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (idx + 1) % 10 == 0:
                if is_main_process():
                    Loss_logger.meters['reg loss'].append((idx + epoch * len(train_loader), metric_logger.meters['reg loss'].avg))
                    Loss_logger.meters['contrast loss'].append((idx + epoch * len(train_loader), metric_logger.meters['contrast loss'].avg))
                    fig = plt.figure(figsize=(8, 6))
                    fig.tight_layout()
                    for (key, value) in Loss_logger.meters.items():
                        plt.plot(*zip(*value), label=key, linewidth=1, markersize=2)
                        plt.legend(loc='upper right', shadow=True, fontsize='medium')
                        plt.grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                        plt.grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    plt.xlabel('Iter')
                    plt.minorticks_on()
                    filename = os.path.join(args.directory, 'training_logger.png')
                    plt.savefig(filename)
                    plt.close()

        with torch.no_grad():
            model.eval()
            avg_neg_distance = val_loader.dataset.create_epoch_tuples(model_without_ddp, device)
            Val_Loss_logger.meters['negative distance'].append((epoch * len(val_loader), avg_neg_distance))
            for idx, (images, features) in enumerate(val_metric_logger.log_every(val_loader, print_freq, '>> Val Epoch: [{}]'.format(epoch))):
                images = images.to(device, non_blocking=True)
                features = features.to(device, non_blocking=True)
                contrast, reg = model(images, features, args.margin)
                loss = contrast + reg

                if not math.isfinite(contrast.item()):
                    print(">> Contrast loss is nan, skipping")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    continue

                if not math.isfinite(reg.item()):
                    print(">> Reg loss is nan, skipping")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()
                val_metric_logger.meters['reg loss'].update(reg.item())
                val_metric_logger.meters['contrast loss'].update(contrast.item())

                if (idx + 1) % 10 == 0:
                    if is_main_process():
                        Val_Loss_logger.meters['reg loss'].append(val_metric_logger.meters['reg loss'].avg)
                        Val_Loss_logger.meters.meters['contrast loss'].append(val_metric_logger.meters['contrast loss'].avg)
                        fig = plt.figure(figsize=(8, 6))
                        fig.tight_layout()
                        for (key, value) in Val_Loss_logger.meters.items():
                            plt.plot(value, 'o-', label=key, linewidth=1, markersize=2)
                        plt.legend(loc='upper right', shadow=True, fontsize='medium')
                        plt.grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                        plt.grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                        plt.set_xlabel('iter')
                        plt.set_ylabel("loss")
                        plt.minorticks_on()
                        plt.savefig(os.path.join(args.directory, 'val_logger.png'))
                        plt.close()

        if val_metric_logger.meters['contrast loss'].avg < min_loss:
            min_loss = val_metric_logger.meters['contrast loss'].avg
            if is_main_process():
                model_path = os.path.join(args.directory, 'best_checkpoint.pth')
                torch.save({'epoch': epoch + 1, 'state_dict': model_without_ddp.state_dict()}, model_path)

        if is_main_process():
            model_path = os.path.join(args.directory, 'epoch{}.pth'.format(epoch + 1))
            model_dict = model_without_ddp.state_dict()
            del model_dict['teacher']
            torch.save({'epoch': epoch + 1, 'state_dict': model_dict}, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='retrieval-SfM-120k', help='training dataset')
    parser.add_argument('--network', type=str, default='mobilenet_v2')
    parser.add_argument('--imsize', default=1024, type=int, metavar='N', help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--resume', default=None, type=str, metavar='FILENAME', help='name of the latest checkpoint (default: None)')
    parser.add_argument('--margin', type=float, default=0.7)

    parser.add_argument('--warmup-epochs', type=int, default=0, help='learning rate will be linearly scaled during warm up period')
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--warmup-lr', type=float, default=0, help='Initial warmup learning rate')
    parser.add_argument('--base-lr', type=float, default=1e-6)
    parser.add_argument('--final-lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29324')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clip_max_norm', type=float, default=0)
    args = parser.parse_args()
    main(args=args)
