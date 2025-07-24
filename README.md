# 微多普勒时频图数据增广项目 - 最小改动版本

基于VA-VAE (Vision foundation model Aligned Variational AutoEncoder) 的微多普勒时频图数据增广项目。本项目在原始LightningDiT项目基础上进行**最小化修改**，实现用户条件化的时频图生成，专门适配31个用户的步态微多普勒时频图数据。

## 🎯 项目特点

- **最小改动**: 仅4个核心文件，基于原项目架构
- **完全兼容**: 使用PyTorch Lightning，与原项目多GPU方式一致  
- **即用性强**: 支持Kaggle数据集结构 (ID_1, ID_2...ID_31)
- **实验可靠**: 最少修改确保结果可信和可对比

## 📁 项目结构

```
VA-VAE/
├── 核心代码文件 (仅4个)
│   ├── minimal_micro_doppler_dataset.py    # 数据加载器
│   ├── minimal_vavae_modification.py       # 用户条件化VA-VAE模型
│   ├── minimal_training_modification.py    # PyTorch Lightning训练脚本
│   └── data_split.py                       # 数据集划分工具
│
├── 配置和文档
│   ├── README.md                           # 项目说明 (本文件)
│   ├── requirements.txt                    # 依赖包列表
│   └── .gitignore                         # Git忽略文件
│
├── 原始项目 (未修改)
│   └── LightningDiT/                      # 完整的原始VA-VAE项目
│
└── 数据目录
    └── data/                              # 数据存储目录
        ├── raw/                           # 原始数据
        ├── processed/                     # 处理后数据
        └── generated/                     # 生成数据
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- CUDA 11.8+ (可选，用于GPU加速)

### 安装依赖
```bash
git clone git@github.com:heimaoqqq/VA-VAE.git
cd VA-VAE
pip install -r requirements.txt
```

### Step 1: 数据集划分
```bash
# 针对Kaggle数据集结构 (ID_1, ID_2...ID_31)
python data_split.py \
    --input_dir /kaggle/input/dataset \
    --output_dir data_split \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_extensions png,jpg,jpeg

# 划分后的结构：
# data_split/
# ├── train/
# │   ├── user_01_sample_001.png
# │   └── ...
# ├── val/
# │   ├── user_01_sample_010.png
# │   └── ...
# └── split_info.txt
```

### Step 2: 模型训练

#### 单GPU训练
```bash
python minimal_training_modification.py \
    --data_dir data_split \
    --original_vavae path/to/vavae.pth \
    --batch_size 16 \
    --max_epochs 100 \
    --devices 1
```

#### 多GPU训练 (推荐)
```bash
# 双GPU训练
python minimal_training_modification.py \
    --data_dir data_split \
    --original_vavae path/to/vavae.pth \
    --batch_size 16 \
    --max_epochs 100 \
    --devices 2 \
    --strategy ddp

# 四GPU训练
python minimal_training_modification.py \
    --data_dir data_split \
    --original_vavae path/to/vavae.pth \
    --batch_size 16 \
    --max_epochs 100 \
    --devices 4 \
    --strategy ddp_find_unused_parameters_true \
    --precision 16  # 混合精度训练
```

#### Kaggle环境训练
```bash
python minimal_training_modification.py \
    --data_dir /kaggle/working/data_split \
    --original_vavae /kaggle/input/pretrained/vavae.pth \
    --batch_size 12 \
    --max_epochs 50 \
    --devices 1 \
    --precision 16 \
    --output_dir /kaggle/working/outputs
```

## 🔧 详细配置说明

### 训练参数
- `--data_dir`: 数据目录路径
- `--original_vavae`: 原始VA-VAE模型路径
- `--batch_size`: 每个GPU的批次大小 (默认16)
- `--max_epochs`: 最大训练轮数 (默认100)
- `--lr`: 学习率 (默认1e-4)
- `--condition_dim`: 用户条件向量维度 (默认128)
- `--kl_weight`: KL散度损失权重 (默认1e-6)

### PyTorch Lightning参数
- `--devices`: GPU数量 (1, 2, 4, 8等)
- `--strategy`: 训练策略 (auto, ddp, ddp_find_unused_parameters_true)
- `--accelerator`: 加速器类型 (gpu, cpu)
- `--precision`: 精度 (16, 32, bf16)

### 数据划分参数
- `--train_ratio`: 训练集比例 (默认0.8)
- `--val_ratio`: 验证集比例 (默认0.2)
- `--image_extensions`: 图像文件扩展名 (默认png,jpg,jpeg)

## 📊 技术架构

### 原项目多GPU支持
本项目使用与原VA-VAE项目相同的多GPU实现方式：
- **PyTorch Lightning**: 自动处理分布式训练
- **DDP策略**: DistributedDataParallel进行数据并行
- **自动同步**: 损失值和梯度自动在GPU间同步

### 用户条件化扩展
- **用户嵌入**: 将用户ID映射到高维特征空间
- **条件注入**: 通过特征相加的方式注入用户信息
- **最小修改**: 在原VA-VAE基础上仅添加必要的条件功能

### 数据处理
- **自动识别**: 支持Kaggle的ID_1...ID_31目录结构
- **格式支持**: PNG, JPG, JPEG等图像格式
- **尺寸标准化**: 自动调整到256×256分辨率
- **RGB转换**: 自动转换为3通道RGB格式

## 📈 性能预期

### 单GPU vs 多GPU
| 配置 | 批次大小 | 训练速度 | 内存使用 | 总吞吐量 |
|------|----------|----------|----------|----------|
| 1×RTX 3080 | 16 | 3秒/批次 | 12GB | 5.3 样本/秒 |
| 2×RTX 3080 | 16×2 | 3秒/批次 | 6GB×2 | 10.6 样本/秒 |

### Kaggle环境
- **GPU**: 通常提供单个GPU (P100/T4)
- **批次大小**: 推荐12-16
- **训练时间**: 约2-4小时 (50 epochs)
- **内存使用**: 8-12GB

## 🔍 监控和日志

### TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir outputs/lightning_logs

# 在浏览器中查看训练进度
# http://localhost:6006
```

### 检查点管理
- **最佳模型**: `outputs/checkpoints/best-*.ckpt`
- **最后模型**: `outputs/checkpoints/last.ckpt`
- **训练日志**: `outputs/lightning_logs/`

## ⚠️ 常见问题

### Q1: CUDA内存不足
```bash
# 解决方案：减小批次大小或使用混合精度
python minimal_training_modification.py --batch_size 8 --precision 16
```

### Q2: 数据加载错误
```bash
# 检查数据文件格式和命名
ls your_data_dir/*.png | head -5

# 确保目录结构为 ID_1/, ID_2/, ..., ID_31/
```

### Q3: 多GPU训练失败
```bash
# 检查GPU数量
nvidia-smi

# 使用推荐的策略
python minimal_training_modification.py --devices 2 --strategy ddp_find_unused_parameters_true
```

## 🎯 与原项目的对比

| 项目 | 文件数量 | 代码行数 | 多GPU方案 | 复杂度 |
|------|----------|----------|-----------|--------|
| 原始LightningDiT | 50+ | 10000+ | HuggingFace Accelerator | 高 |
| 原始VA-VAE | 30+ | 8000+ | PyTorch Lightning | 中 |
| **我们的版本** | **4** | **<600** | **PyTorch Lightning** | **低** |

## 📝 开发历程

- [x] 项目初始化和环境搭建
- [x] 微多普勒数据预处理模块
- [x] 条件VA-VAE模型设计
- [x] 训练策略优化
- [x] 模型训练和验证
- [x] 生成接口开发
- [x] PyTorch Lightning集成
- [x] 多GPU支持完善
- [x] Kaggle环境适配

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目基于原始LightningDiT项目，遵循相应的开源许可证。

## 🙏 致谢

- 感谢LightningDiT项目提供的VA-VAE基础架构
- 感谢PyTorch Lightning团队提供的优秀框架
- 感谢开源社区的贡献和支持

---

**项目地址**: https://github.com/heimaoqqq/VA-VAE  
**问题反馈**: 请在GitHub Issues中提出问题和建议
