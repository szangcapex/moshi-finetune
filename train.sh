#!/bin/bash
# chmod +x

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="./output_${TIMESTAMP}"

echo "Training will save to: $RUN_DIR"

# 删除旧的符号链接（如果存在）
if [ -L "./output" ]; then
    rm ./output
fi

# 创建新的符号链接指向本次训练目录
ln -s "output_${TIMESTAMP}" ./output

echo "Created symlink: ./output -> output_${TIMESTAMP}"

CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml