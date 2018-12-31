#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python test_thyroid.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Thyroid" \
    --num_mlp_layer 1 \
    --recur_steps 5 \
    --model_path "recurrent/5/global-epoch-152-acc-0.896.pth"
