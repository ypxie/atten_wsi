#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "pool" \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --model_path "pool/865.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --model_path "global/NW1/884.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 2 \
    --use_w_loss False \
    --model_path "global/NW2/876.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "global/W1/888.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 2 \
    --use_w_loss True \
    --model_path "global/W2/876.pth"


CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "self" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --model_path "self/NW1/884.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "self" \
    --dataset "Thyroid" \
    --num_mlp_layer 2 \
    --use_w_loss False \
    --model_path "self/NW2/876.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "self" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "self/W1/884.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "self" \
    --dataset "Thyroid" \
    --num_mlp_layer 2 \
    --use_w_loss True \
    --model_path "self/W2/876.pth"
