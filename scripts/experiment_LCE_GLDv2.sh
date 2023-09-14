#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 --use_env ../LCE_training.py \
--network mobilenet_v2 \
--imsize 512 \
--batch_size 128 \
--num-workers 8 \
--device cuda \
--num_epochs 20 \
--warmup-epochs 5 \
--warmup-lr 0.0001 \
--base-lr 0.001 \
--final-lr 0 \
--momentum 0.9 \
--weight-decay 0.000001 \
--clip_max_norm -1.0 \
--seed 11 \












