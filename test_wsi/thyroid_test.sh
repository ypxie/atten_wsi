#!/bin/bash

CUDA_VISIBLE_DEVICES=2 \
python test_thyroid.py \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "W1/global-epoch-299-acc-0.888.pth"

CUDA_VISIBLE_DEVICES=2 \
python test_thyroid.py \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "W1/global-epoch-296-acc-0.888.pth"


CUDA_VISIBLE_DEVICES=2 \
python test_thyroid.py \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "W1/global-epoch-292-acc-0.888.pth"

CUDA_VISIBLE_DEVICES=2 \
python test_thyroid.py \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "W1/global-epoch-289-acc-0.888.pth"


CUDA_VISIBLE_DEVICES=2 \
python test_thyroid.py \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "W1/global-epoch-284-acc-0.888.pth"

CUDA_VISIBLE_DEVICES=2 \
python test_thyroid.py \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "W1/global-epoch-270-acc-0.888.pth"
