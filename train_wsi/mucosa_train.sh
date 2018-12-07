#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python train_mucosa.py \
    --fea_mix "self" \
    --dataset "Mucosa" \
    --pre_load True \
    --num_mlp_layer 2 \
    --use_w_loss False \
    --maxepoch 300 \
    --session 1


# use_w_loss:False num_mlp_layer:1 session: 0
# use_w_loss:False num_mlp_layer:2 session: 1
# use_w_loss:True  num_mlp_layer:1 session: 2
# use_w_loss:True  num_mlp_layer:2 session: 3
