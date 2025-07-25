# VA-VAE 微多普勒信号生成项目

基于LightningDiT的微多普勒信号图像生成项目，使用VA-VAE进行特征提取和DiT进行条件化生成。

## ⭐ 官方方法：严格按照LightningDiT README

**完全按照官方README步骤执行：**

### 使用方法
```bash
python setup_official_models.py
```

**自动执行官方步骤**：
1. 📥 下载预训练模型 (VA-VAE + LightningDiT-XL + latents_stats.pt)
2. ⚙️ 修改配置文件 (基于官方reproductions配置)
3. 🔧 更新VA-VAE配置路径 (官方教程要求)
4. 🚀 运行 `bash run_fast_inference.sh` (官方推理脚本)

**输出**: `LightningDiT/demo_images/demo_samples.png`

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- Accelerate
- CUDA (推荐)

## 📁 项目结构

```
VA-VAE/
├── setup_official_models.py        # 官方方法一键执行
├── LightningDiT/                   # 官方LightningDiT项目
└── official_models/                # 下载的预训练模型
```

## 📖 技术说明

基于LightningDiT官方预训练模型，实现ImageNet-256级别的高质量图像生成 (FID=1.35)。

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
