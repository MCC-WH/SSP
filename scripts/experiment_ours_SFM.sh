#!/bin/bash

python ../Ours_training.py \
--dataset retrieval-SfM-120k \
--network mobilenet_v2 \
--m 32 \
--n_bits 8 \
--anchor 1024 \
--imsize 362 \
--batch_size 64 \
--num-workers 4 \
--device cuda \
--num_epochs 10 \
--warmup-epochs 5 \
--warmup-lr 0.0001 \
--base-lr 0.001 \
--final-lr 0 \
--momentum 0.9 \
--weight-decay 0.000001 \
--clip_max_norm -1.0 \
--seed 11 \












