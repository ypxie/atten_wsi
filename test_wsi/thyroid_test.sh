#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python test_thyroid_global.py \
    --model_name "global" \
    --dataset "Thyroid" \
    --model_path "880/global-epoch-250-acc-0.880.pth"
