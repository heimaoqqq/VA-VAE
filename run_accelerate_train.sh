#!/bin/bash

# 基于Accelerate的多GPU训练脚本
# 参考LightningDiT原项目的训练方式

echo "🎯 启动基于Accelerate的多GPU DiT训练"
echo "================================================"

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "🔧 检测到 $GPU_COUNT 个GPU"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

# 使用accelerate launch启动训练
accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port 29500 \
    stage2_train_dit_accelerate.py \
    --latent_dir /kaggle/working/latent_features \
    --output_dir /kaggle/working/trained_models_accelerate \
    --batch_size 16 \
    --max_epochs 100 \
    --lr 1e-4 \
    --seed 42

echo "✅ 训练完成!"
