#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python train_thyroid_global.py \
    --model_name "global" \
    --dataset "Thyroid" \
    --pre_load True \
    --maxepoch 300 \
    --session 0
