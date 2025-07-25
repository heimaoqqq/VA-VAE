# VA-VAE 微多普勒信号生成项目

基于LightningDiT的微多普勒信号图像生成项目，使用VA-VAE进行特征提取和DiT进行条件化生成。

## ⭐ 推荐方法：严格按照官方方法

**我们完全按照LightningDiT官方README和教程实现，确保最佳效果：**

### 🎯 官方方法对照表

| 步骤 | 官方要求 | 我们的实现 | 状态 |
|------|----------|------------|------|
| 模型下载 | 手动下载3个文件 | `setup_official_models.py` | ✅ 自动化 |
| 配置修改 | 修改reproductions配置 | 自动生成标准配置 | ✅ 标准化 |
| VA-VAE配置 | 修改tokenizer配置路径 | 自动更新路径 | ✅ 自动化 |
| 推理命令 | `bash run_fast_inference.sh` | 封装在Python脚本中 | ✅ 简化 |

### 1. 下载官方预训练模型
```bash
python setup_official_models.py
```
**下载内容**：
- ✅ VA-VAE: `vavae-imagenet256-f16d32-dinov2.pt` (官方tokenizer)
- ✅ LightningDiT-XL: `lightningdit-xl-imagenet256-800ep.pt` (FID=1.35性能)
- ✅ 潜在统计: `latents_stats.pt` (数据归一化)

### 2. 验证设置正确性
```bash
python verify_official_setup.py
```
**检查项目**：
- 📁 模型文件完整性
- ⚙️ 配置参数正确性
- 🔧 VA-VAE路径设置
- 📋 与官方配置对比

### 3. 测试模型加载
```bash
python test_official_models.py
```

### 4. 运行官方推理
```bash
python run_official_inference.py
```
**技术细节**：
- 🎯 使用官方 `bash run_fast_inference.sh`
- 🎨 Demo模式: cfg_scale=9.0 (自动覆盖配置)
- 📊 250采样步数，Euler方法
- 🖼️ 输出: `LightningDiT/demo_images/demo_samples.png`

---

## 🎯 项目特点

- **三阶段流水线**: 特征提取 → DiT训练 → 图像生成
- **环境检查**: 内置环境和依赖检查功能
- **错误修复**: 解决了导入、颜色、维度等关键问题
- **即用性强**: 支持Kaggle和本地环境

## 📁 项目结构

```
VA-VAE/
├── 核心流水线 (3个阶段)
│   ├── stage1_extract_features.py    # 阶段1: VA-VAE特征提取 (含环境检查)
│   ├── stage2_train_dit.py          # 阶段2: DiT模型训练 (含环境检查)
│   └── stage3_inference.py          # 阶段3: 图像生成推理 (含导入测试)
│
├── 辅助工具
│   ├── data_split.py               # 数据集划分工具
│   └── download_vavae_model.py     # VA-VAE模型下载
│
├── 配置文件
│   ├── vavae_config.yaml          # VA-VAE配置
│   ├── accelerate_config.yaml     # Accelerate分布式配置
│   └── requirements.txt           # 依赖包列表
│
├── 原始项目
│   └── LightningDiT/              # 完整的LightningDiT项目
│
└── 数据目录
    └── data/                      # 数据存储目录
        ├── raw/                   # 原始数据
        ├── processed/             # 处理后数据
        └── generated/             # 生成数据
```

## 🚀 快速开始

### 环境检查
```bash
# 1. 测试导入是否正常
python stage3_inference.py --test_imports

# 2. 检查特征提取环境
python stage1_extract_features.py --help

# 3. 检查训练环境
python stage2_train_dit.py --help
```

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (GPU推理)
- Accelerate (分布式训练)
- Safetensors (模型保存)

### 安装依赖
```bash
git clone git@github.com:heimaoqqq/VA-VAE.git
cd VA-VAE
pip install -r requirements.txt
```

## 🎯 完整流水线 (推荐使用)

### 一键运行
```bash
# 运行完整流水线 (特征提取 + 训练 + 生成)
python complete_pipeline.py

# 自定义训练轮数和输出目录
python complete_pipeline.py --max_epochs 100 --output_dir ./my_samples

# 只生成样本 (跳过训练，使用现有模型)
python complete_pipeline.py --skip_extract --skip_train

# 强制重新训练模型
python complete_pipeline.py --force_retrain --max_epochs 100
```

### 流水线参数说明
- `--max_epochs`: 训练轮数 (默认50)
- `--checkpoint_dir`: 模型保存目录 (默认./checkpoints)
- `--output_dir`: 生成样本输出目录 (默认./generated_samples)
- `--skip_extract`: 跳过特征提取
- `--skip_train`: 跳过模型训练
- `--skip_generate`: 跳过样本生成
- `--force_retrain`: 强制重新训练模型

## 📋 分步执行 (高级用户)

### 阶段1: 特征提取
```bash
python stage1_extract_features.py
```

### 阶段2: 模型训练
```bash
python stage2_train_dit.py \
    --latent_dir ./data/processed \
    --output_dir ./checkpoints \
    --max_epochs 50 \
    --batch_size 16
```

### 阶段3: 样本生成
```bash
python stage3_inference.py \
    --dit_checkpoint ./checkpoints/best_model \
    --vavae_config vavae_config.yaml \
    --output_dir ./generated_samples \
    --user_ids 1 2 3 4 5 \
    --num_samples_per_user 4
```

## ⚠️ 常见问题和解决方案

### 问题1: 生成的图像质量很差 (像噪声)
**原因**: 模型没有正确加载训练好的权重，使用的是随机初始化的模型

**解决方案**:
1. 确保训练完成并保存了检查点:
   ```bash
   # 检查是否有训练好的模型
   ls -la checkpoints/best_model/
   ```

2. 如果没有检查点，重新训练:
   ```bash
   python complete_pipeline.py --force_retrain --max_epochs 100
   ```

3. 如果有检查点但仍然质量差，增加训练轮数:
   ```bash
   python complete_pipeline.py --force_retrain --max_epochs 200
   ```

### 问题2: 训练过程中出现错误
**常见错误和解决方案**:
- `CUDA out of memory`: 减少batch_size
- `模块导入错误`: 检查依赖安装
- `数据加载错误`: 检查数据目录结构

### 问题3: 特征提取失败
**解决方案**:
1. 检查VA-VAE模型是否正确下载:
   ```bash
   python verify_vavae_setup.py
   ```

2. 重新下载VA-VAE模型:
   ```bash
   python download_vavae_model.py
   ```

## 📊 性能优化建议

### 训练优化
- 使用更大的batch_size (如果GPU内存允许)
- 增加训练轮数到100-200轮
- 使用学习率调度器

### 生成优化
- 调整guidance_scale (2.0-8.0)
- 增加采样步数 (250-1000)
- 尝试不同的用户ID

## 🔧 原始Kaggle方法 (已弃用)

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
