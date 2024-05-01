#!/bin/bash
python3 train.py \
    --accelerator gpu \
    --devices 1 \
    --batch_size 256 \
    --num_datas 50000 \
    --max_epochs 10 \
    --lr 0.01 \
    --num_heads 8 \
    --dim 512
