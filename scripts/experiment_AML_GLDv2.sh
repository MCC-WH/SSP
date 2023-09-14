#!/bin/bash

python ../AML_training.py \
--dataset GLDv2 \
--network mobilenet_v2 \
--imsize 512 \
--margin 0.7 \
--batch_size 128 \
--num-workers 8 \
--device cuda \
--num_epochs 200 \
--warmup-epochs 10 \
--warmup-lr 0.0001 \
--base-lr 0.001 \
--final-lr 0 \
--momentum 0.9 \
--weight-decay 0.000001 \
--clip_max_norm -1.0 \
--seed 11 \












