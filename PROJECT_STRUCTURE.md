# 项目结构说明

## 📁 最小改动版本文件结构

```
VA-VAE/
├── README.md                           # 项目主要说明文档
├── PROJECT_SUMMARY.md                  # 项目完成总结
├── PROJECT_STRUCTURE.md               # 项目结构说明 (本文件)
├── requirements.txt                    # 依赖包列表
│
├── 核心代码文件 (仅4个)
├── minimal_micro_doppler_dataset.py    # 数据加载器
├── minimal_vavae_modification.py       # 用户条件化VA-VAE模型
├── minimal_training_modification.py    # PyTorch Lightning训练脚本
├── data_split.py                       # 数据集划分工具
│
├── 使用说明文档
├── MINIMAL_USAGE.md                    # 最小版本使用说明
├── LIGHTNING_USAGE.md                  # PyTorch Lightning版本详细说明
├── KAGGLE_USAGE.md                     # Kaggle环境使用说明
├── minimal_adaptation.md               # 最小改动方案说明
│
├── 原始项目
├── LightningDiT/                       # 原始VA-VAE项目 (未修改)
│   ├── train.py                        # 原始训练脚本
│   ├── tokenizer/vavae.py             # 原始VA-VAE模型
│   ├── vavae/                         # VA-VAE相关代码
│   └── ...                            # 其他原始文件
│
└── 数据目录 (运行时创建)
    ├── data/                          # 数据存储目录
    │   ├── raw/                       # 原始数据
    │   ├── processed/                 # 处理后数据
    │   └── generated/                 # 生成数据
    └── outputs/                       # 训练输出目录
```

## 🎯 核心文件说明

### 1. 数据处理
- **`data_split.py`**: 将Kaggle数据集(ID_1, ID_2...ID_31)按用户划分为训练集和验证集
- **`minimal_micro_doppler_dataset.py`**: 数据加载器，支持原始结构和划分后结构

### 2. 模型定义
- **`minimal_vavae_modification.py`**: 在原始VA-VAE基础上添加用户条件功能

### 3. 训练脚本
- **`minimal_training_modification.py`**: 使用PyTorch Lightning的训练脚本，支持多GPU

## 🚀 使用流程

### 快速开始
```bash
# 1. 数据划分
python data_split.py --input_dir /kaggle/input/dataset --output_dir data_split

# 2. 训练模型
python minimal_training_modification.py --data_dir data_split --original_vavae path/to/vavae.pth

# 3. 多GPU训练
python minimal_training_modification.py --data_dir data_split --original_vavae path/to/vavae.pth --devices 2 --strategy ddp
```

## 📊 改动量统计

| 项目 | 文件数量 | 代码行数 | 复杂度 |
|------|----------|----------|--------|
| 原始项目 | 50+ | 10000+ | 高 |
| **最小改动版本** | **4** | **<600** | **低** |

## 🎯 设计原则

1. **最小修改**: 只添加必要的用户条件功能
2. **保持兼容**: 使用原项目的多GPU方式 (PyTorch Lightning)
3. **易于理解**: 代码简洁，逻辑清晰
4. **实验可靠**: 基于原始架构，结果可信

## 📝 文档说明

- **README.md**: 项目概述和基本使用方法
- **MINIMAL_USAGE.md**: 最小版本的详细使用说明
- **LIGHTNING_USAGE.md**: PyTorch Lightning版本的完整说明
- **KAGGLE_USAGE.md**: 针对Kaggle环境的特殊说明
- **PROJECT_SUMMARY.md**: 项目开发过程和成果总结

## ⚠️ 注意事项

1. **原始项目**: LightningDiT目录包含完整的原始项目，未做任何修改
2. **依赖管理**: requirements.txt包含所有必要依赖，包括pytorch-lightning
3. **数据格式**: 支持Kaggle数据集的ID_1...ID_31目录结构
4. **模型加载**: 需要提供预训练的VA-VAE模型文件

这个项目结构确保了最小化修改的同时，提供了完整的微多普勒时频图条件生成功能！
