"""
Kaggle运行脚本 - 启动Step6 DiT训练
在Kaggle notebook中运行此脚本
"""

import os
import sys
from pathlib import Path

# 设置环境
print("=" * 50)
print("Step 6: 训练LightningDiT with VA-VAE")
print("=" * 50)

# 1. 克隆和设置代码库
print("\n1. 设置代码库...")
os.system("cd /kaggle/working && git clone https://github.com/FoundationVision/LightningDiT.git")
os.system("cd /kaggle/working && git clone https://github.com/heimaoqqq/VA-VAE.git")

# 2. 安装依赖
print("\n2. 安装依赖...")
os.system("pip install -q omegaconf einops timm")

# 3. 下载预训练权重
print("\n3. 下载预训练权重...")
weights_dir = Path("/kaggle/working/weights")
weights_dir.mkdir(exist_ok=True)

# 下载LightningDiT-B权重
os.system(f"""
cd {weights_dir} && \
wget -q https://huggingface.co/Maple/LightningDiT/resolve/main/LightningDiT-B-2-256.pth
""")

# 下载VA-VAE权重（使用微调后的）
vavae_weight = Path("/kaggle/working/VA-VAE/outputs/vavae_finetuned_best.pth")
if not vavae_weight.exists():
    print("警告: 未找到微调后的VA-VAE权重，请先运行step4训练VA-VAE")

# 4. 准备数据
print("\n4. 准备数据...")
data_dir = Path("/kaggle/working/microdoppler_data")
if not data_dir.exists():
    print("准备micro-Doppler数据集...")
    # 这里假设数据已经准备好或从Kaggle dataset加载

# 5. 复制训练脚本
print("\n5. 复制训练脚本...")
os.system("cp /kaggle/working/VA-VAE/microdoppler_finetune/step6_train_dit.py /kaggle/working/")

# 6. 运行训练
print("\n6. 启动训练...")
print("-" * 30)

# 检测GPU数量
import torch
n_gpus = torch.cuda.device_count()
print(f"检测到 {n_gpus} 个GPU")

if n_gpus > 0:
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 运行训练脚本
train_cmd = f"""
cd /kaggle/working && \
python step6_train_dit.py
"""

print(f"\n执行命令:\n{train_cmd}")
os.system(train_cmd)

print("\n" + "=" * 50)
print("训练完成！")
print("=" * 50)
