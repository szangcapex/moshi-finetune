#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 如果 output 目录存在（不是符号链接），重命名它
if [ -e "./output" ] && [ ! -L "./output" ]; then
    OLD_NAME="./output_archived_${TIMESTAMP}"
    echo "Found existing output directory, renaming to: $OLD_NAME"
    mv ./output "$OLD_NAME"
fi

CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml