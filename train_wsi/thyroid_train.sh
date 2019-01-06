#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python train_thyroid.py \
    --pre_load \
    --use_w_loss \
    --patch_mix "att" \
    --fea_mix "global" \
    --recur_steps 5 \
    --num_mlp_layer 1 \
    --maxepoch 500 \
    --session 5
