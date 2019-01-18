#!/bin/bash

CUDA_VISIBLE_DEVICES=5 \
python train_mucosa.py \
    --pre_load \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --num_mlp_layer 1 \
    --recur_steps 4 \
    --maxepoch 100 \
    --session 4 \
    --batch_size 16
