#!/bin/bash
python3 train_lightning.py \
    --accelerator gpu \
    --devices 1 \
    --batch_size 128 \
    --max_epochs 30 \
    --lr 0.001 \
    --num_heads 8 \
    --dim 512
