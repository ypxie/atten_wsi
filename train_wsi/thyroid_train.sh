#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
python train_thyroid.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --use_w_loss False \
    --num_mlp_layer 1 \
    --pre_load True \
    --recur_steps 6 \
    --maxepoch 500 \
    --session 6
