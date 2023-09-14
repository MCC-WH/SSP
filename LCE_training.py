import math
import os
import argparse
from networks import efficientnet_b3, mobilenet_v2
from utils.helpfunc import get_checkpoint_root, get_data_root
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import cuda, optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
from Dataset import GLDv2_CL
from networks import LCE
from utils import MetricLogger, create_optimizer, init_distributed_mode, is_main_process, get_rank, optimizer_to, NumberLogger


def collect_batch(batch):
    batch = list(filter(lambda x: x is not None, batch))
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(targets, dim=0)


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.reshape(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]


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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    args.directory = os.path.join(get_checkpoint_root(), 'GLDv2-{}-LCE'.format(args.network))
    os.makedirs(args.directory, exist_ok=True)

    if args.distributed:
        ngpus_per_node = cuda.device_count()
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print('>> batch size per node:{}'.format(args.batch_size))
        print('>> num workers per node:{}'.format(args.num_workers))
    csv_path = os.path.join(get_data_root(), 'train', 'GLDv2', 'train.csv')
    clean_csv_path = os.path.join(get_data_root(), 'train', 'GLDv2', 'train_clean.csv')
    image_dir = os.path.join(get_data_root(), 'train', 'GLDv2', 'train')
    output_directory = os.path.join(get_data_root(), 'train', 'GLDv2')
    train_dataset, val_dataset, class_num = GLDv2_CL(csv_path, clean_csv_path, image_dir, output_directory, imsize=args.imsize)
    args.classifier_num = class_num

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=False)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=True)
        val_sampler = DistributedSampler(val_dataset)
        val_batch_sampler = BatchSampler(val_sampler, args.batch_size, drop_last=False)
        val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True)

    if args.network == 'mobilenet_v2':
        backbone = mobilenet_v2(2048)
    elif args.network == 'efficientnet_b3':
        backbone = efficientnet_b3(2048)
    else:
        raise ValueError('Unsupport backbone type')
    model = LCE(backbone=backbone, reduction_dim=2048, classifier_num=args.classifier_num).to(device)
    model_without_ddp = model

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('>> number of params:{:.2f}M'.format(n_parameters / (1024 * 1024)))

    # define optimizer
    param_dicts = create_optimizer(args.weight_decay, model_without_ddp)
    optimizer = optim.SGD(param_dicts, lr=args.base_lr * args.batch_size / 256, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True, dampening=0.0)
    scaler = GradScaler()

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optim'])
            optimizer_to(optimizer, device)
            scaler.load_state_dict(checkpoint['scaler'])
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
    Loss_logger = NumberLogger()
    Error_Logger = NumberLogger()
    Val_Loss_logger = NumberLogger()
    Val_Error_Logger = NumberLogger()
    min_loss = 10000.0

    for epoch in range(start_epoch, args.num_epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch + 1 + get_rank())
        header = '>> Train Epoch: [{}]'.format(epoch)
        optimizer.zero_grad()
        for idx, (images, targets) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            model.train()
            targets = targets.to(device, non_blocking=True)
            loss, logits, loss_old, logits_old, weight_consistance = model(images.to(device, non_blocking=True), targets)
            total_loss = loss + 0.5 * loss_old + 10 * weight_consistance

            if not math.isfinite(loss.item()):
                print(">> ArcFace loss is nan, skipping")
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            if not math.isfinite(loss_old.item()):
                print(">> Old arcFace loss is nan, skipping")
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            if not math.isfinite(weight_consistance.item()):
                print(">> Weight consistance is nan, skipping")
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                continue

            scaler.scale(total_loss).backward()
            metric_logger.meters['ArcFace loss'].update(loss.item())
            metric_logger.meters['ArcFace old loss'].update(loss_old.item())
            metric_logger.meters['Weight loss'].update(10 * weight_consistance.item())
            with torch.no_grad():
                desc_top1_err, desc_top5_err = topk_errors(logits, targets, [1, 5])
                metric_logger.meters['Top1 error'].update(desc_top1_err.item())
                metric_logger.meters['Top5 error'].update(desc_top5_err.item())
                desc_top1_err, desc_top5_err = topk_errors(logits_old, targets, [1, 5])
                metric_logger.meters['Top1 old error'].update(desc_top1_err.item())
                metric_logger.meters['Top5 old error'].update(desc_top5_err.item())

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
                    Loss_logger.meters['ArcFace loss'].append(metric_logger.meters['ArcFace loss'].avg)
                    Loss_logger.meters['ArcFace old loss'].append(metric_logger.meters['ArcFace old loss'].avg)
                    Loss_logger.meters['Weight loss'].append(metric_logger.meters['Weight loss'].avg)
                    Error_Logger.meters['Top1 error'].append(metric_logger.meters['Top1 error'].avg)
                    Error_Logger.meters['Top5 error'].append(metric_logger.meters['Top5 error'].avg)
                    Error_Logger.meters['Top1 old error'].append(metric_logger.meters['Top1 old error'].avg)
                    Error_Logger.meters['Top5 old error'].append(metric_logger.meters['Top5 old error'].avg)
                    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
                    fig.tight_layout()
                    axes = axes.flatten()
                    for (key, value) in Loss_logger.meters.items():
                        axes[0].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[0].legend(loc='upper right', shadow=True, fontsize='medium')
                    axes[0].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[0].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[0].set_xlabel('iter')
                    axes[0].set_ylabel("loss")
                    axes[0].minorticks_on()
                    for (key, value) in Error_Logger.meters.items():
                        axes[1].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[1].legend(loc='upper right', shadow=True, fontsize='medium')
                    axes[1].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[1].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[1].set_xlabel('iter')
                    axes[1].set_ylabel("Error rate (%)")
                    axes[1].minorticks_on()
                    plt.savefig(os.path.join(args.directory, 'training_logger.png'))
                    plt.close()

        with torch.no_grad():
            model.eval()
            for idx, (inputs, labels) in enumerate(val_metric_logger.log_every(val_loader, print_freq, '>> Val Epoch: [{}]'.format(epoch))):
                inputs, labels = inputs.to(device), labels.to(device, non_blocking=True)
                loss, logits, loss_old, logits_old, weight_consistance = model(inputs, labels)

                val_metric_logger.meters['ArcFace loss'].update(loss.item())
                val_metric_logger.meters['ArcFace old loss'].update(loss_old.item())
                val_metric_logger.meters['Weight loss'].update(10 * weight_consistance.item())
                with torch.no_grad():
                    desc_top1_err, desc_top5_err = topk_errors(logits, targets, [1, 5])
                    val_metric_logger.meters['Top1 error'].update(desc_top1_err.item())
                    val_metric_logger.meters['Top5 error'].update(desc_top5_err.item())
                    desc_top1_err, desc_top5_err = topk_errors(logits_old, targets, [1, 5])
                    val_metric_logger.meters['Top1 old error'].update(desc_top1_err.item())
                    val_metric_logger.meters['Top5 old error'].update(desc_top5_err.item())

                if (idx + 1) % 10 == 0:
                    if is_main_process():
                        Val_Loss_logger.meters['ArcFace loss'].append(val_metric_logger.meters['ArcFace loss'].avg)
                        Val_Loss_logger.meters.meters['ArcFace old loss'].append(val_metric_logger.meters['ArcFace old loss'].avg)
                        Val_Loss_logger.meters.meters['Weight loss'].append(val_metric_logger.meters['Weight loss'].avg)
                        Val_Error_Logger.meters['Top1 error'].append(val_metric_logger.meters['Top1 error'].avg)
                        Val_Error_Logger.meters['Top5 error'].append(val_metric_logger.meters['Top5 error'].avg)
                        Val_Error_Logger.meters['Top1 old error'].append(val_metric_logger.meters['Top1 old error'].avg)
                        Val_Error_Logger.meters['Top5 old error'].append(val_metric_logger.meters['Top5 old error'].avg)
                        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
                        fig.tight_layout()
                        axes = axes.flatten()
                        for (key, value) in Val_Loss_logger.meters.items():
                            axes[0].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                        axes[0].legend(loc='upper right', shadow=True, fontsize='medium')
                        axes[0].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                        axes[0].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                        axes[0].set_xlabel('iter')
                        axes[0].set_ylabel("loss")
                        axes[0].minorticks_on()
                        for (key, value) in Val_Error_Logger.meters.items():
                            axes[1].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                        axes[1].legend(loc='upper right', shadow=True, fontsize='medium')
                        axes[1].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                        axes[1].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                        axes[1].set_xlabel('iter')
                        axes[1].set_ylabel("Error rate (%)")
                        axes[1].minorticks_on()
                        plt.savefig(os.path.join(args.directory, 'val_logger.png'))
                        plt.close()

        if val_metric_logger.meters['ArcFace old loss'].avg < min_loss:
            min_loss = val_metric_logger.meters['ArcFace old loss'].avg
            if is_main_process():
                model_path = os.path.join(args.directory, 'best_checkpoint.pth')
                torch.save({'epoch': epoch + 1, 'state_dict': model_without_ddp.state_dict()}, model_path)

        if is_main_process():
            # Save checkpoint
            model_path = os.path.join(args.directory, 'epoch{}.pth'.format(epoch + 1))
            torch.save({'epoch': epoch + 1, 'state_dict': model_without_ddp.state_dict(), 'optim': optimizer.state_dict(), 'scaler': scaler.state_dict()}, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_num', type=int, default=1000)
    parser.add_argument('--network', type=str, default='mobilenet_v2')
    parser.add_argument('--imsize', default=1024, type=int, metavar='N', help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--resume', default=None, type=str, metavar='FILENAME', help='name of the latest checkpoint (default: None)')

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
