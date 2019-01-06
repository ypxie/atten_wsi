#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --model_path "global/NW1/759.pth"
