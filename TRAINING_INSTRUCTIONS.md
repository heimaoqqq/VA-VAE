# 🎯 微多普勒时频图数据增广完整训练指令

## 📋 **完整的7步训练流程**

### **环境准备阶段**

#### **Step 1: 克隆项目**
```bash
git clone https://github.com/heimaoqqq/VA-VAE.git
cd VA-VAE
```

#### **Step 2: 一键安装所有依赖** ✨
```bash
# 安装完整依赖 (推理 + VA-VAE训练 + Taming-Transformers)
python install_dependencies.py

# 或者只安装推理依赖
python install_dependencies.py --inference-only
```

**包含的依赖：**
- ✅ LightningDiT基础推理依赖
- ✅ VA-VAE训练专用依赖 (pytorch-lightning, lpips, kornia等)
- ✅ Taming-Transformers自动安装和torch 2.x兼容性修复
- ✅ 微多普勒训练所需的所有包

#### **Step 2.5: 下载预训练模型**
```bash
# 微多普勒训练专用 (强烈推荐，仅800MB)
python step1_download_models.py --vae-only

# 或完整下载 (包含不必要的6GB推理模型)
python step1_download_models.py
```

**模型说明：**
- ✅ **VA-VAE模型** (~800MB) - 微调训练的基础，**必需**
- ❌ **潜在统计** (~1KB) - ImageNet统计，**不适用**微多普勒数据
- ❌ **扩散模型** (~6GB) - ImageNet 1000类，**不适用**31用户条件

**为什么选择 --vae-only？**
1. **latents_stats.pt**: 基于ImageNet计算，对微多普勒数据不准确
2. **LightningDiT-XL**: 1000类ImageNet vs 31个用户，架构不匹配
3. **重头训练**: 我们会训练专门的31用户条件扩散模型

---

### **数据准备阶段**

#### **Step 3: 准备数据集**
```bash
# 确保您的数据集结构如下：
# your_micro_doppler_data/
# ├── ID_1/  (用户1的150张时频图)
# ├── ID_2/  (用户2的150张时频图)
# └── ... ID_31/ (用户31的150张时频图)

# 运行数据集划分和准备 (8:2比例)
python step3_prepare_micro_doppler_dataset.py \
    --input_dir /path/to/your_micro_doppler_data \
    --output_dir micro_doppler_dataset

# 验证数据集结构
ls micro_doppler_dataset/
# 应该看到：
# ├── train/ (31个用户，每用户约120张)
# │   ├── user1/ user2/ ... user31/
# ├── val/ (31个用户，每用户约30张)
# │   ├── user1/ user2/ ... user31/
# ├── dataset_config.yaml
# └── split_info.txt
```

---

### **VA-VAE微调阶段**

#### **Step 4: VA-VAE微调** (2-3天)
```bash
# 启动VA-VAE微调，针对T4×2 GPU优化
python step4_finetune_vavae.py \
    --dataset_dir micro_doppler_dataset \
    --output_dir vavae_finetuned

# 监控训练进度
tensorboard --logdir LightningDiT/vavae/logs/

# 训练完成后，checkpoint保存在：
# LightningDiT/vavae/logs/[timestamp]/checkpoints/last.ckpt
```

**关键配置：**
- 学习率：1e-5 (微调)
- 批次大小：1 (T4×2优化)
- 梯度累积：4
- 训练轮数：100轮
- 使用方案B增强用户嵌入

---

### **特征提取阶段**

#### **Step 5: 提取潜在特征** (2-3小时)
```bash
# 使用微调后的VA-VAE提取32×16×16潜在特征
python step5_extract_latent_features.py \
    --dataset_dir micro_doppler_dataset \
    --checkpoint_path LightningDiT/vavae/logs/[实际时间戳]/checkpoints/last.ckpt \
    --output_dir micro_doppler_latents

# 验证特征提取结果
ls micro_doppler_latents/
# 应该看到：
# ├── train/
# │   ├── user1/ user2/ ... user31/ (每个包含.pt特征文件)
# ├── val/
# │   ├── user1/ user2/ ... user31/
# └── latent_dataset_config.yaml
```

---

### **扩散模型训练阶段**

#### **Step 6: 训练LightningDiT扩散模型** (1-2周)
```bash
# 训练扩散模型，支持31个用户条件生成
python step6_train_diffusion_model.py \
    --latent_dir micro_doppler_latents \
    --output_dir lightningdit_trained \
    --num_users 31 \
    --batch_size 2

# 监控训练
tensorboard --logdir lightningdit_trained/logs/
```

---

### **生成测试阶段**

#### **Step 7: 生成微多普勒图像** (10分钟)
```bash
# 生成指定用户的时频图
python step7_generate_micro_doppler.py \
    --diffusion_model lightningdit_trained/checkpoints/best.ckpt \
    --vae_model LightningDiT/vavae/logs/[时间戳]/checkpoints/last.ckpt \
    --user_id 5 \
    --num_samples 10 \
    --output_dir generated_samples

# 批量生成所有用户 (数据增广)
for user_id in {1..31}; do
    python step7_generate_micro_doppler.py \
        --diffusion_model lightningdit_trained/checkpoints/best.ckpt \
        --vae_model LightningDiT/vavae/logs/[时间戳]/checkpoints/last.ckpt \
        --user_id $user_id \
        --num_samples 50 \
        --output_dir generated_samples/user_$user_id
done
```

---

## ⏱️ **时间和资源预估**

| 阶段 | 步骤 | 预计时间 | GPU使用 | 关键输出 |
|------|------|----------|---------|----------|
| 环境准备 | Step 1-2 | 30分钟 | - | 完整依赖环境 |
| 数据准备 | Step 3 | 10分钟 | - | 训练/验证数据集 |
| VA-VAE微调 | Step 4 | **2-3天** | T4×2 | 微调VA-VAE模型 |
| 特征提取 | Step 5 | 2-3小时 | T4×1 | 潜在特征数据集 |
| 扩散训练 | Step 6 | **1-2周** | T4×2 | 条件扩散模型 |
| 生成测试 | Step 7 | 10分钟 | T4×1 | 增广时频图 |

---

## 🔍 **关键检查点**

### **Step 3 完成检查**
```bash
cat micro_doppler_dataset/split_info.txt
# 应该显示31个用户的8:2训练/验证分布
```

### **Step 4 完成检查**
```bash
# 检查VA-VAE微调效果
python -c "
import torch
ckpt_path = 'LightningDiT/vavae/logs/[时间戳]/checkpoints/last.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
print('VA-VAE微调完成，epoch:', ckpt.get('epoch', 'unknown'))
print('验证损失:', ckpt.get('val_loss', 'unknown'))
"
```

### **Step 5 完成检查**
```bash
# 检查潜在特征
python -c "
import torch
from pathlib import Path
latent_files = list(Path('micro_doppler_latents/train/user1').glob('*.pt'))
if latent_files:
    sample = torch.load(latent_files[0])
    print('潜在特征形状:', sample['latent'].shape)  # 应该是 [32, 16, 16]
    print('用户ID:', sample['user_id'])  # 应该是 0-30
else:
    print('未找到潜在特征文件')
"
```

---

## 🚨 **故障排除**

### **内存不足**
```bash
# 在step4_finetune_vavae.py中调整：
# batch_size: 1 → batch_size: 1 (已经是最小)
# accumulate_grad_batches: 4 → accumulate_grad_batches: 8
```

### **训练中断恢复**
```bash
# VA-VAE训练恢复
python step4_finetune_vavae.py \
    --dataset_dir micro_doppler_dataset \
    --resume_from_checkpoint /path/to/checkpoint

# 扩散模型训练恢复
python step6_train_diffusion_model.py \
    --latent_dir micro_doppler_latents \
    --resume_from_checkpoint /path/to/checkpoint
```

---

## 🎯 **最终目标**

完成训练后，您将获得：
1. **微调的VA-VAE** - 专门编码/解码微多普勒时频图
2. **条件扩散模型** - 可指定用户ID生成该用户的新时频图
3. **数据增广能力** - 为每个用户生成大量高质量的新样本
4. **用户特征保持** - 生成的图像保持指定用户的步态特征

这样就完成了微多普勒时频图的数据增广系统！🚀
