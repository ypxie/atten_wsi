#!/bin/bash

CUDA_VISIBLE_DEVICES=4 \
python train_thyroid.py \
    --pre_load \
    --patch_mix "att" \
    --fea_mix "global" \
    --num_mlp_layer 1 \
    --recur_steps 4 \
    --maxepoch 100 \
    --session 4 \
    --batch_size 16

    # --use_w_loss \
