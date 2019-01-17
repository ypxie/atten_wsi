#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python train_thyroid.py \
    --pre_load \
    --patch_mix "att" \
    --fea_mix "global" \
    --num_mlp_layer 1 \
    --recur_steps 1 \
    --maxepoch 300 \
    --session 0

    # --use_w_loss \
