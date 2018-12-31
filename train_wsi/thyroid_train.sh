#!/bin/bash

CUDA_VISIBLE_DEVICES=2 \
python train_thyroid.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --recur_steps 5 \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --pre_load True \
    --maxepoch 500 \
    --session 5
