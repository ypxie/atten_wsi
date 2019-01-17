#!/bin/bash

CUDA_VISIBLE_DEVICES=2 \
python test_mucosa.py \
    --patch_mix "att" \
    --fea_mix "global" \
    --dataset "Mucosa" \
    --num_mlp_layer 1 \
    --recur_steps 5 \
    --model_path "recurrent/5/833.pth"


    # --pre_load \
