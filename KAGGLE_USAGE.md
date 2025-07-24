# Kaggle数据集使用说明

## 🎯 针对您的数据集结构

### 📁 您的数据集结构
```
/kaggle/input/dataset/
├── ID_1/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── ID_2/
│   ├── image_001.png
│   └── ...
...
└── ID_31/
    ├── image_001.png
    └── ...
```

## 📊 关于测试集的建议

### 🔍 原项目分析结果
- **LightningDiT**: 使用Accelerator + DDP，主要用于生成任务
- **VA-VAE**: 使用pytorch-lightning，通常只有train/val划分
- **建议**: 对于条件生成任务，**只使用训练集和验证集即可**

### ✅ 推荐的数据划分
```bash
# 推荐：只划分训练集和验证集 (8:2)
python data_split.py \
    --input_dir /kaggle/input/dataset \
    --output_dir /kaggle/working/data_split \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_extensions png,jpg,jpeg

# 如果坚持要测试集 (7:2:1)
python data_split.py \
    --input_dir /kaggle/input/dataset \
    --output_dir /kaggle/working/data_split \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --use_test_set \
    --test_ratio 0.1 \
    --image_extensions png,jpg,jpeg
```

## 🚀 完整使用流程

### Step 1: 数据集划分
```bash
# 在Kaggle环境中运行
python data_split.py \
    --input_dir /kaggle/input/dataset \
    --output_dir /kaggle/working/data_split \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --seed 42

# 划分后的结构：
# /kaggle/working/data_split/
# ├── train/
# │   ├── user_01_sample_001.png
# │   ├── user_01_sample_002.png
# │   ├── user_02_sample_001.png
# │   └── ...
# ├── val/
# │   ├── user_01_sample_010.png
# │   ├── user_02_sample_008.png
# │   └── ...
# └── split_info.txt
```

### Step 2: 单GPU训练（Kaggle默认）
```bash
python minimal_training_modification.py \
    --data_dir /kaggle/working/data_split \
    --original_vavae /path/to/vavae.pth \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir /kaggle/working/outputs
```

### Step 3: 双GPU训练（如果Kaggle提供）
```bash
# 检查GPU数量
nvidia-smi

# 如果有2个GPU
python minimal_training_modification.py \
    --data_dir /kaggle/working/data_split \
    --original_vavae /path/to/vavae.pth \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir /kaggle/working/outputs \
    --distributed \
    --world_size 2
```

## 🔧 关于多GPU支持

### 📋 原项目vs我们的实现

| 项目 | 多GPU方案 | 优势 | 劣势 |
|------|-----------|------|------|
| **LightningDiT** | HuggingFace Accelerator | 简单易用，自动处理 | 依赖额外库 |
| **VA-VAE** | PyTorch Lightning | 功能丰富，易配置 | 框架重量级 |
| **我们的实现** | 原生PyTorch DDP | 轻量级，可控性强 | 需要手动配置 |

### ✅ 我们的多GPU实现优势
1. **最小依赖**: 只使用PyTorch原生功能
2. **完全兼容**: 与原VA-VAE架构完全兼容
3. **损失同步**: 正确处理分布式训练中的损失传递
4. **内存效率**: 每个GPU处理不同数据子集

## 🎯 针对您数据集的特殊处理

### 1. 自动用户ID映射
- **ID_1** → **user_01**
- **ID_2** → **user_02**
- ...
- **ID_31** → **user_31**

### 2. 图像格式支持
- 支持 PNG, JPG, JPEG 等常见格式
- 自动转换为RGB格式（LightningDiT要求）
- 自动调整尺寸到256×256

### 3. 按用户划分
- 确保每个用户在训练集和验证集中都有样本
- 保持用户内样本的时序关系（如果重要）

## ⚠️ Kaggle环境注意事项

### 1. 存储限制
```bash
# 检查磁盘空间
df -h /kaggle/working

# 如果空间不足，可以减少输出文件
# 或者直接在原始数据上训练（不推荐）
```

### 2. 内存限制
```bash
# Kaggle通常提供16GB RAM
# 如果内存不足，减少batch_size
python minimal_training_modification.py --batch_size 8
```

### 3. GPU限制
```bash
# Kaggle通常提供单个GPU (P100/T4)
# 检查GPU类型和内存
nvidia-smi
```

### 4. 时间限制
```bash
# Kaggle有运行时间限制
# 建议设置较少的epochs进行测试
python minimal_training_modification.py --epochs 50
```

## 🔍 验证数据加载

### 测试数据集类
```python
# 在Kaggle notebook中测试
from minimal_micro_doppler_dataset import MicroDopplerDataset

# 测试原始结构
dataset = MicroDopplerDataset(
    '/kaggle/input/dataset', 
    original_structure=True
)
print(f"原始数据集大小: {len(dataset)}")
print(f"用户数量: {dataset.num_users}")

# 测试划分后结构
train_dataset = MicroDopplerDataset(
    '/kaggle/working/data_split', 
    split='train'
)
print(f"训练集大小: {len(train_dataset)}")

# 测试加载一个样本
sample = dataset[0]
print(f"样本形状: {sample['image'].shape}")
print(f"用户ID: {sample['user_id']}")
```

## 📈 预期性能

### Kaggle P100 GPU
- **批次大小**: 16-24
- **训练速度**: ~3-4秒/批次
- **内存使用**: ~10-12GB

### Kaggle T4 GPU  
- **批次大小**: 12-16
- **训练速度**: ~2-3秒/批次
- **内存使用**: ~8-10GB

## 🎯 实验建议

### 1. 快速验证
```bash
# 先用小数据集验证流程
python data_split.py --input_dir /kaggle/input/dataset --output_dir /kaggle/working/test_split --train_ratio 0.9 --val_ratio 0.1

# 短时间训练测试
python minimal_training_modification.py --epochs 5 --batch_size 8
```

### 2. 正式训练
```bash
# 确认流程无误后进行正式训练
python minimal_training_modification.py --epochs 100 --batch_size 16
```

### 3. 结果保存
```bash
# 将重要结果保存到Kaggle输出
cp /kaggle/working/outputs/best_model.pth /kaggle/working/
cp /kaggle/working/data_split/split_info.txt /kaggle/working/
```

这样您就可以在Kaggle环境中高效地使用我们的最小改动版本进行实验了！
