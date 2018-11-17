#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python test_mucosa_global.py \
    --model_name "global" \
    --dataset "Mucosa" \
    --model_path "704/global-epoch-188-acc-0.741.pth"
