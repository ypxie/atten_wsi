#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
python train_mucosa_global.py \
    --model_name "global" \
    --dataset "Mucosa" \
    --pre_load True \
    --maxepoch 300 \
    --session 0
