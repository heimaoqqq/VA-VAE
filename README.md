# 微多普勒时频图数据增广项目 (Micro-Doppler Spectrogram Data Augmentation)

基于VA-VAE (Vision foundation model Aligned Variational AutoEncoder) 的微多普勒时频图数据增广解决方案

## 🎯 项目目标

- 解决微多普勒时频图数据量不足的问题
- 实现指定用户的步态微多普勒时频图生成
- 处理31个用户间差异较小的数据集
- 提供高质量的数据增广方案

## 📊 数据特点

- **数据类型**: 步态微多普勒时频图
- **用户数量**: 31个用户
- **数据特征**: 用户间差异较小，数据量有限
- **目标**: 条件生成指定用户的时频图

## 🏗️ 技术架构

### 核心组件
1. **条件VA-VAE**: 基于用户条件的变分自编码器
2. **时频图预处理**: 专门的微多普勒数据处理模块
3. **用户编码器**: 用户特征编码和条件注入
4. **生成接口**: 指定用户的时频图生成API

### 技术优势
- ✅ 小数据集友好的训练策略
- ✅ 21倍训练加速 (基于VA-VAE)
- ✅ 条件生成特定用户数据
- ✅ 高质量时频图重建

## 📁 项目结构

```
micro_doppler_vavae/
├── data/                          # 数据目录
│   ├── raw/                       # 原始微多普勒数据
│   ├── processed/                 # 预处理后的数据
│   └── generated/                 # 生成的增广数据
├── src/                           # 源代码
│   ├── models/                    # 模型定义
│   │   ├── conditional_vavae.py   # 条件VA-VAE模型
│   │   ├── user_encoder.py        # 用户编码器
│   │   └── discriminator.py       # 判别器(可选)
│   ├── data/                      # 数据处理
│   │   ├── micro_doppler_dataset.py  # 数据集类
│   │   ├── preprocessing.py       # 预处理工具
│   │   └── augmentation.py        # 数据增强
│   ├── training/                  # 训练相关
│   │   ├── trainer.py             # 训练器
│   │   ├── losses.py              # 损失函数
│   │   └── metrics.py             # 评估指标
│   └── utils/                     # 工具函数
│       ├── visualization.py       # 可视化工具
│       ├── evaluation.py          # 评估工具
│       └── config.py              # 配置管理
├── configs/                       # 配置文件
│   ├── model_config.yaml          # 模型配置
│   ├── training_config.yaml       # 训练配置
│   └── data_config.yaml           # 数据配置
├── scripts/                       # 脚本文件
│   ├── train.py                   # 训练脚本
│   ├── inference.py               # 推理脚本
│   ├── evaluate.py                # 评估脚本
│   └── generate_data.py           # 数据生成脚本
├── notebooks/                     # Jupyter notebooks
│   ├── data_exploration.ipynb     # 数据探索
│   ├── model_analysis.ipynb       # 模型分析
│   └── results_visualization.ipynb # 结果可视化
├── tests/                         # 测试文件
├── requirements.txt               # 依赖包
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 环境安装
```bash
# 创建conda环境
conda create -n micro_doppler_vavae python=3.10
conda activate micro_doppler_vavae

# 安装依赖
pip install -r requirements.txt
```

### 数据准备
```bash
# 预处理微多普勒数据
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed --create_stats --visualize

# 数据探索
jupyter notebook notebooks/data_exploration.ipynb
```

### 模型训练
```bash
# 训练条件VA-VAE
python scripts/train.py --config configs/training_config.yaml --data_dir data/processed

# 恢复训练（可选）
python scripts/train.py --config configs/training_config.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

### 模型评估
```bash
# 全面评估
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/processed --eval_all

# 单项评估
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --eval_reconstruction
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --eval_generation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --eval_user_specificity
```

### 数据生成
```bash
# 生成指定用户的时频图
python scripts/inference.py --checkpoint checkpoints/best_model.pth --user_id 1 --num_samples 5

# 批量生成数据增广样本
python scripts/generate_data.py --checkpoint checkpoints/best_model.pth --num_samples_per_user 50 --create_manifest

# 指定用户批量生成
python scripts/generate_data.py --checkpoint checkpoints/best_model.pth --user_ids "0,1,2,5,10" --num_samples_per_user 20
```

### Web界面使用
```bash
# 启动Web界面
python scripts/web_interface.py --checkpoint checkpoints/best_model.pth --host 0.0.0.0 --port 5000

# 访问 http://localhost:5000 使用图形界面生成时频图
```

## 📈 预期效果

- **重建质量**: 高保真度的时频图重建
- **用户特异性**: 准确生成指定用户的特征
- **数据多样性**: 丰富的变化和细节
- **训练效率**: 快速收敛，适合小数据集

## 🔧 核心特性

### 1. 条件生成
- 用户ID条件注入
- 用户特征编码
- 可控的生成过程

### 2. 小数据集优化
- 数据增强策略
- 正则化技术
- 迁移学习

### 3. 质量保证
- 多层次损失函数
- 感知损失
- 用户一致性约束

## 📊 评估指标

- **重建质量**: PSNR, SSIM, LPIPS
- **用户特异性**: 分类准确率, 特征相似度
- **数据多样性**: FID, IS, 特征分布
- **时频特性**: 频谱保真度, 时间一致性

## 🎯 应用场景

- 步态识别数据增广
- 雷达信号处理
- 生物特征识别
- 运动分析研究

## 📝 开发计划

- [x] 项目初始化和环境搭建
- [x] 微多普勒数据预处理模块
- [x] 条件VA-VAE模型设计
- [x] 训练策略优化
- [x] 模型训练和验证
- [x] 生成接口开发

## 🔧 详细使用说明

### 配置文件说明

项目提供了三个主要配置文件：

1. **模型配置** (`configs/model_config.yaml`)
   - 模型架构参数
   - 用户编码器设置
   - 损失函数权重

2. **训练配置** (`configs/training_config.yaml`)
   - 训练超参数
   - 优化器设置
   - 数据增强策略

3. **数据配置** (`configs/data_config.yaml`)
   - 数据路径和格式
   - 预处理参数
   - 时频图特定设置

### 数据格式要求

支持的数据格式：
- **NumPy格式** (`.npy`): 推荐格式，加载速度快
- **HDF5格式** (`.h5`): 适合大数据集
- **图像格式** (`.png`, `.jpg`): 便于可视化

文件命名规范：
```
user_{user_id}_sample_{sample_id}.npy
# 例如: user_01_sample_001.npy, user_02_sample_015.npy
```

### 训练技巧

1. **小数据集优化**
   ```bash
   # 使用更强的数据增强
   python scripts/train.py --config configs/training_config.yaml --augment_prob 0.8

   # 调整学习率
   python scripts/train.py --learning_rate 5e-5 --epochs 200
   ```

2. **多GPU训练**
   ```bash
   # 使用多GPU加速训练
   CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --config configs/training_config.yaml
   ```

3. **调试模式**
   ```bash
   # 快速验证代码
   python scripts/train.py --debug --epochs 2 --batch_size 8
   ```

### 生成质量控制

1. **温度参数调节**
   ```bash
   # 更随机的生成 (temperature > 1.0)
   python scripts/generate_data.py --temperature 1.2 --num_samples_per_user 10

   # 更确定性的生成 (temperature < 1.0)
   python scripts/generate_data.py --temperature 0.8 --num_samples_per_user 10
   ```

2. **种子控制**
   ```bash
   # 可重现的生成
   python scripts/inference.py --seed 42 --user_id 5 --num_samples 3
   ```

### 性能优化建议

1. **内存优化**
   - 减少batch_size如果遇到OOM错误
   - 使用混合精度训练 (`use_amp: true`)
   - 启用梯度检查点

2. **训练加速**
   - 使用预训练的视觉基础模型
   - 调整数据加载器的num_workers
   - 启用模型编译 (PyTorch 2.0+)

### 常见问题解决

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   python scripts/train.py --batch_size 16

   # 启用梯度累积
   python scripts/train.py --gradient_accumulation_steps 2
   ```

2. **训练不收敛**
   ```bash
   # 降低学习率
   python scripts/train.py --learning_rate 1e-5

   # 增加warmup轮数
   python scripts/train.py --warmup_epochs 20
   ```

3. **生成质量差**
   ```bash
   # 增加训练轮数
   python scripts/train.py --epochs 300

   # 调整损失权重
   python scripts/train.py --kl_weight 1e-5 --align_weight 2.0
   ```

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目基于MIT许可证开源。

## 🙏 致谢

本项目基于以下优秀工作：
- [LightningDiT](https://github.com/hustvl/LightningDiT) - VA-VAE原始实现
- [DiT](https://github.com/facebookresearch/DiT) - Diffusion Transformer
- [LDM](https://github.com/CompVis/latent-diffusion) - Latent Diffusion Models
