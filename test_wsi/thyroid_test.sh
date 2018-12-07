#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
python test_thyroid.py \
    --patch_mix "pool" \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --use_w_loss True \
    --model_path "pool/global-epoch-245-acc-0.865.pth"
