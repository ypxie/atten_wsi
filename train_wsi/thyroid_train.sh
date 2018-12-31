#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python train_thyroid.py \
    --pre_load \
    --patch_mix "att" \
    --fea_mix "global" \
    --recur_steps 4 \
    --num_mlp_layer 1 \
    --maxepoch 300 \
    --session 4

    # --use_w_loss \
