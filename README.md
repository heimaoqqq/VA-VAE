# LightningDiT 官方复现指南 (Kaggle环境)

## 🎯 项目概述

这是基于 [LightningDiT](https://github.com/hustvl/LightningDiT) 的官方复现项目，专门针对Kaggle环境优化。

### 核心技术
- **VA-VAE**: Vision Foundation Model Aligned Variational AutoEncoder
- **LightningDiT**: Lightning Diffusion Transformer (Transformer-based扩散模型)
- **性能**: ImageNet-256 FID=1.35 (SOTA)

## 🚀 Kaggle环境完整复现指令

### 前置要求
- Kaggle GPU环境 (P100/T4/V100)
- 至少15GB可用磁盘空间
- 稳定的网络连接

### 分步执行指令

#### 步骤1: 环境安装
```bash
!python step1_install_environment.py
```
**功能**: 安装LightningDiT官方依赖，适配Kaggle环境
**时间**: 5-10分钟
**输出**: 环境验证报告

#### 步骤2: 模型下载  
```bash
!python step2_download_models.py
```
**功能**: 下载预训练模型 (VA-VAE + LightningDiT + 统计文件)
**时间**: 10-30分钟 (约7GB)
**输出**: models/ 目录包含3个模型文件

#### 步骤3: 配置设置
```bash
!python step3_setup_configs.py
```
**功能**: 设置推理配置，适配Kaggle路径
**时间**: 1-2分钟
**输出**: kaggle_inference_config.yaml

#### 步骤4A: 运行推理（标准版）
```bash
!python step4_run_inference.py
```
**功能**: 执行LightningDiT推理生成ImageNet图像
**时间**: 10-20分钟
**输出**: LightningDiT/demo_images/demo_samples.png

#### 步骤4B: 运行推理（简化版，推荐）
```bash
!python step4_demo_inference.py
```
**功能**: 简化版demo推理，避免路径和数据集问题
**时间**: 5-15分钟
**输出**: LightningDiT/demo_output/
**推荐**: 如果步骤4A失败，使用此版本

#### 故障排除: 修复latents_stats.pt文件（如果需要）
```bash
!python fix_latents_stats.py
```
**功能**: 修复空的或损坏的latents_stats.pt文件
**使用场景**: 如果出现 "torch.cat(): expected a non-empty list of Tensors" 错误

## 📁 项目结构

```
VA-VAE/
├── LightningDiT/                           # 官方LightningDiT项目
│   ├── models/                             # 模型架构
│   ├── tokenizer/                          # VA-VAE实现
│   ├── configs/                            # 配置文件
│   ├── inference.py                        # 推理脚本
│   └── demo_images/                        # 输出图像
├── models/                                 # 下载的预训练模型
│   ├── vavae-imagenet256-f16d32-dinov2.pt # VA-VAE模型 (~800MB)
│   ├── lightningdit-xl-imagenet256-800ep.pt # DiT模型 (~6GB)
│   └── latents_stats.pt                   # 统计文件 (~1KB)
├── step1_install_environment.py           # 环境安装脚本
├── step2_download_models.py               # 模型下载脚本
├── step3_setup_configs.py                 # 配置设置脚本
├── step4_run_inference.py                 # 推理执行脚本
├── kaggle_inference_config.yaml           # Kaggle适配配置
└── README.md                              # 本文档
```

## 🔧 技术架构

### 1. VA-VAE (主要创新)
- **基础**: 与LDM VAE相同的CNN架构
- **创新**: 与DINOv2视觉基础模型对齐
- **效果**: 解决传统VAE的优化困境
- **编码**: 256×256 → 32×16×16 (f16d32)

### 2. LightningDiT (Transformer扩散模型)
- **架构**: 纯Transformer，非UNet
- **特点**: 28层Transformer + AdaLN调制
- **改进**: RMSNorm + QK归一化 + SwiGLU + RoPE
- **性能**: 21.8× 训练加速

### 3. Transport采样器
- **方法**: ODE求解器
- **特点**: 分类器自由引导 (CFG)
- **配置**: 50步采样 (Kaggle优化)

## 📊 预期结果

### 成功标志
- ✅ 所有步骤无错误执行
- ✅ 生成高质量ImageNet图像
- ✅ 输出文件: `LightningDiT/demo_images/demo_samples.png`
- ✅ 图像包含1000个ImageNet类别的生成样本

### 性能指标
- **生成质量**: FID=1.35 (ImageNet-256 SOTA)
- **推理时间**: 10-20分钟 (50步采样)
- **内存需求**: ~8GB GPU内存
- **模型大小**: VA-VAE ~800MB, DiT ~6GB

## 🐛 常见问题

### 环境问题
**问题**: PyTorch版本冲突
**解决**: 脚本会自动安装官方指定版本 (torch==2.2.0)

**问题**: TorchDiffEq安装失败
**解决**: 脚本包含多种安装策略，通常可自动解决

### 下载问题
**问题**: 模型下载缓慢
**解决**: 使用稳定网络，脚本支持断点续传

**问题**: HuggingFace连接失败
**解决**: 检查网络或使用VPN

### 推理问题
**问题**: GPU内存不足
**解决**: 脚本已优化批次大小 (batch_size=2)

**问题**: 推理超时
**解决**: 脚本设置30分钟超时，通常足够

## 🎯 下一步: 适配微多普勒数据

复现成功后，可以考虑：

1. **数据适配**: 将31用户微多普勒数据适配到VA-VAE
2. **模型选择**: 
   - 继续使用Transformer (需要数据增强)
   - 或改用UNet扩散模型 (更适合小数据集)
3. **训练策略**: 微调vs从头训练

## 📚 参考资料

- [LightningDiT官方论文](https://arxiv.org/abs/2412.09958)
- [LightningDiT GitHub](https://github.com/hustvl/LightningDiT)
- [VA-VAE技术细节](https://github.com/hustvl/LightningDiT/tree/main/vavae)

## 🤝 支持

如遇问题，请检查：
1. 每个步骤的输出日志
2. 文件和目录是否正确创建
3. 网络连接是否稳定
4. GPU内存是否充足
