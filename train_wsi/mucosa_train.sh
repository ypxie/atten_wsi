#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python train_mucosa.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --pre_load True \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --maxepoch 300 \
    --session 1
