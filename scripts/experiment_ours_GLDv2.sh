#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 --use_env ../Ours_training.py \
--dataset GLDv2 \
--network mobilenet_v2 \
--m 32 \
--n_bits 8 \
--anchor 1024 \
--imsize 512 \
--batch_size 128 \
--num-workers 8 \
--device cuda \
--num_epochs 5 \
--warmup-epochs 2 \
--warmup-lr 0.0001 \
--base-lr 0.001 \
--final-lr 0 \
--momentum 0.9 \
--weight-decay 0.000001 \
--clip_max_norm -1.0 \
--seed 11 \

























