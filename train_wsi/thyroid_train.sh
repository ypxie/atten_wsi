#!/bin/bash

CUDA_VISIBLE_DEVICES=4 \
python train_thyroid.py \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --use_w_loss False \
    --num_mlp_layer 1 \
    --pre_load True \
    --maxepoch 300 \
    --session 0

# use_w_loss:False num_mlp_layer:1 session: 0
# use_w_loss:False num_mlp_layer:2 session: 1
# use_w_loss:True  num_mlp_layer:1 session: 2
# use_w_loss:True  num_mlp_layer:2 session: 3
