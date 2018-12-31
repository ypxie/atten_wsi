#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
python train_thyroid.py \
    --dataset "Thyroid" \
    --use_w_loss False \
    --num_mlp_layer 1 \
    --patch_mix "att" \
    --fea_mix "global" \
    --pre_load True \
    --maxepoch 500 \
    --recur_steps 5 \
    --session 5
