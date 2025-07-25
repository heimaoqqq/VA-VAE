# VA-VAE 微多普勒信号生成项目

基于LightningDiT的微多普勒信号图像生成项目，使用VA-VAE进行特征提取和DiT进行条件化生成。

## ⭐ 官方方法：严格按照LightningDiT README

### 🚀 一键执行 (推荐)
```bash
python setup_official_models.py
```

### 📋 分步执行 (详细控制)

**步骤1: 下载预训练模型**
```bash
python step1_download_models.py
```

**步骤2: 设置配置文件**
```bash
python step2_setup_configs.py
```

**步骤3: 运行推理**
```bash
python step3_run_inference.py
```

**输出**: `LightningDiT/demo_images/demo_samples.png`

### 环境要求
- Python 3.10+ (官方推荐3.10.12)
- PyTorch 2.2.0 (官方指定版本)
- TorchDiffEq (关键依赖)
- Accelerate, TIMM, Diffusers等 (见官方requirements.txt)

### 依赖安装
```bash
# 严格按照官方requirements.txt安装
python install_dependencies.py
```

## � 快速开始

### 1. 安装依赖
```bash
python install_dependencies.py
```

### 2. 下载模型
```bash
python step1_download_models.py
```

### 3. 设置配置
```bash
python step2_setup_configs.py
```

### 4. 运行推理
```bash
python step3_run_inference.py
```

## �📁 项目结构

```
VA-VAE/
├── setup_official_models.py        # 官方方法一键执行
├── LightningDiT/                   # 官方LightningDiT项目
└── official_models/                # 下载的预训练模型
```

## 📖 技术说明

基于LightningDiT官方预训练模型，实现ImageNet-256级别的高质量图像生成 (FID=1.35)。

### 🔬 扩展资源

**VA-VAE训练代码**：
- 官方训练代码：https://github.com/hustvl/LightningDiT/tree/main/vavae
- 可以训练自定义的VA-VAE模型

**实验性模型变体**：
- HuggingFace仓库：https://huggingface.co/hustvl/va-vae-imagenet256-experimental-variants
- 包含多种配置的预训练模型：
  - `vavae-imagenet256-f16d32-dinov2-50ep.ckpt` (1.57GB)
  - `vavae-imagenet256-f16d32-mae-50ep.ckpt` (1.56GB)
  - `vavae-imagenet256-f16d64-dinov2-50ep.ckpt` (1.57GB)
  - `vavae-imagenet256-f16d64-mae-50ep.ckpt` (1.57GB)
  - `ldm-imagenet256-f16d16-50ep.ckpt` (349MB)
  - `ldm-imagenet256-f16d32-50ep.ckpt` (349MB)
  - `ldm-imagenet256-f16d64-50ep.ckpt` (351MB)

## ⚠️ 常见问题

### 模型下载失败
- 检查网络连接
- 使用代理或VPN
- 手动从HuggingFace下载

### 推理失败
- 确认安装了accelerate: `pip install accelerate`
- 检查CUDA环境
- 确认模型文件完整

## � 相关资源

- **官方论文**: [Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models](https://arxiv.org/abs/2501.01423)
- **官方代码**: [LightningDiT GitHub](https://github.com/hustvl/LightningDiT)
- **预训练模型**: [HuggingFace Models](https://huggingface.co/hustvl)
