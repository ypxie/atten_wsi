#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --model_path "global/NW1/759.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --num_mlp_layer 2 \
    --use_w_loss False \
    --model_path "global/NW2/741.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "global/W1/759.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --num_mlp_layer 2 \
    --use_w_loss True \
    --model_path "global/W2/741.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "pool" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --model_path "pool/741.pth"


CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "self" \
    --dataset "Mucosa" \
    --num_mlp_layer 1 \
    --use_w_loss False \
    --model_path "self/NW1/759.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "self" \
    --dataset "Mucosa" \
    --num_mlp_layer 2 \
    --use_w_loss False \
    --model_path "self/NW2/741.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "self" \
    --dataset "Mucosa" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "self/W1/759.pth"

CUDA_VISIBLE_DEVICES=7 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "self" \
    --dataset "Mucosa" \
    --num_mlp_layer 2 \
    --use_w_loss True \
    --model_path "self/W2/741.pth"
