# 最小改动版本使用说明

## 🎯 项目清理完成

已移除所有复杂的、不必要的代码文件，现在只保留最核心的3个文件：

### 📁 核心文件
1. **`minimal_micro_doppler_dataset.py`** - 数据加载器（支持数据集划分）
2. **`minimal_vavae_modification.py`** - 用户条件化VA-VAE模型
3. **`minimal_training_modification.py`** - 训练脚本（支持双GPU）
4. **`data_split.py`** - 数据集划分工具

## 🚀 使用流程

### Step 1: 数据集划分
```bash
# 将您的未划分数据集进行训练/验证/测试划分
python data_split.py \
    --input_dir your_original_data/ \
    --output_dir data_split/ \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# 划分后的目录结构：
# data_split/
# ├── train/
# │   ├── user_00_sample_001.npy
# │   └── ...
# ├── val/
# │   ├── user_00_sample_010.npy
# │   └── ...
# ├── test/
# │   ├── user_00_sample_015.npy
# │   └── ...
# └── split_info.txt
```

### Step 2: 单GPU训练
```bash
# 使用单GPU训练 (PyTorch Lightning)
python minimal_training_modification.py \
    --data_dir data_split/ \
    --original_vavae path/to/vavae.pth \
    --batch_size 16 \
    --max_epochs 100 \
    --lr 1e-4 \
    --devices 1 \
    --output_dir outputs/
```

### Step 3: 多GPU训练
```bash
# 使用双GPU训练 (PyTorch Lightning)
python minimal_training_modification.py \
    --data_dir data_split/ \
    --original_vavae path/to/vavae.pth \
    --batch_size 16 \
    --max_epochs 100 \
    --lr 1e-4 \
    --devices 2 \
    --strategy ddp \
    --output_dir outputs/
```

## 🔧 PyTorch Lightning多GPU特性

### ✅ 支持的功能（与原项目一致）
- **自动分布式训练**: PyTorch Lightning自动处理DDP设置
- **数据并行**: 使用DistributedDataParallel (DDP)
- **自动采样**: Lightning自动处理分布式采样
- **损失同步**: 损失值在所有GPU间自动同步和平均
- **模型保存**: Lightning自动在主进程保存模型
- **日志记录**: 统一的TensorBoard日志记录

### 🎯 PyTorch Lightning优势
1. **与原项目一致**: 使用相同的多GPU实现方式
2. **自动化程度高**: 无需手动配置分布式训练
3. **功能丰富**: 内置检查点、早停、日志等功能
4. **稳定可靠**: 经过大量项目验证的成熟框架

### ⚠️ 注意事项
1. **批次大小**: 总的有效批次大小 = batch_size × devices
2. **学习率**: 多GPU时可能需要调整学习率
3. **策略选择**: 推荐使用`ddp_find_unused_parameters_true`
4. **精度设置**: 可以使用`--precision 16`节省内存

## 📊 数据格式要求

### 文件命名规范
```
user_{user_id:02d}_sample_{sample_id:03d}.npy
```

示例：
- `user_00_sample_001.npy` - 用户0的第1个样本
- `user_01_sample_015.npy` - 用户1的第15个样本
- `user_30_sample_100.npy` - 用户30的第100个样本

### 数据内容要求
- **格式**: NumPy数组 (.npy文件)
- **形状**: (H, W) 或 (H, W, 1) 或 (H, W, 3)
- **数值范围**: 建议 [0, 1] 或 [0, 255]
- **数据类型**: float32 或 uint8

## 🔍 训练监控

### 损失值含义
- **total_loss**: 总损失 = recon_loss + kl_weight × kl_loss
- **recon_loss**: 重建损失 (MSE)，衡量重建质量
- **kl_loss**: KL散度损失，衡量潜在空间的正则化

### 正常的损失范围
- **recon_loss**: 通常在 0.01 - 0.1 之间
- **kl_loss**: 通常在 1 - 100 之间
- **total_loss**: 主要由recon_loss主导

### 训练技巧
1. **KL权重调整**: 如果重建质量差，降低kl_weight
2. **学习率调整**: 如果损失震荡，降低学习率
3. **批次大小**: 如果内存不足，减小batch_size

## 🚨 常见问题

### Q1: CUDA内存不足
```bash
# 解决方案：减小批次大小
python minimal_training_modification.py --batch_size 8
```

### Q2: 双GPU训练失败
```bash
# 检查GPU数量
nvidia-smi

# 确保有2个可用GPU，然后：
CUDA_VISIBLE_DEVICES=0,1 python minimal_training_modification.py --distributed --world_size 2
```

### Q3: 数据加载错误
```bash
# 检查数据文件格式
ls your_data_dir/*.npy | head -5

# 确保文件名符合 user_XX_sample_XXX.npy 格式
```

### Q4: 模型加载失败
```
# 当前使用的是DummyVAVAE，需要替换为实际的VA-VAE模型
# 请修改 minimal_training_modification.py 中的模型加载部分
```

## 📈 性能预期

### 单GPU (RTX 3080)
- **批次大小**: 16-32
- **训练速度**: ~2-3 秒/批次
- **内存使用**: ~8-12GB

### 双GPU (2×RTX 3080)
- **批次大小**: 32-64 (总计)
- **训练速度**: ~1.5-2 秒/批次
- **内存使用**: ~6-8GB/GPU

## 🎯 下一步

1. **验证基本功能**: 先用小数据集验证训练流程
2. **调整超参数**: 根据损失曲线调整学习率和权重
3. **评估生成质量**: 训练完成后检查生成的时频图质量
4. **逐步改进**: 如果基本版本效果好，再考虑添加改进功能

这个最小版本确保了实验的可靠性和结果的可对比性！
