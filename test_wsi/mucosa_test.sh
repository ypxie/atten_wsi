#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --fea_mix "self" \
    --dataset "Mucosa" \
    --num_mlp_layer 2 \
    --use_w_loss False \
    --model_path "NW2/self-epoch-255-acc-0.759.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --fea_mix "self" \
    --dataset "Mucosa" \
    --num_mlp_layer 2 \
    --use_w_loss False \
    --model_path "NW2/self-epoch-256-acc-0.759.pth"
