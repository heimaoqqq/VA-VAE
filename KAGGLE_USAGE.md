# VA-VAE Kaggle使用指南

本指南专门针对在Kaggle环境中使用VA-VAE进行微调的完整流程。

## 🚀 快速开始

### 第一步：克隆项目
```bash
!git clone https://github.com/heimaoqqq/VA-VAE.git
%cd VA-VAE
```

### 第二步：安装依赖
**重要：请使用我们提供的安装脚本，它会按照官方要求正确安装所有依赖**

```bash
!python install_dependencies.py
```

如果安装脚本失败，请手动执行以下步骤：

#### 手动安装taming-transformers（官方方式）
```bash
# 克隆并安装taming-transformers
!git clone https://github.com/CompVis/taming-transformers.git
%cd taming-transformers
!pip install -e .

# 修复torch 2.x兼容性
!sed -i 's/from torch._six import string_classes/from six import string_types as string_classes/' taming/data/utils.py

%cd ..
```

#### 安装其他依赖
```bash
!pip install pytorch-lightning omegaconf einops transformers accelerate
```

### 第三步：验证安装
```python
# 运行依赖检查
!python -c "
import taming.data.utils as tdu
import taming.modules.losses.vqperceptual
from taming.modules.vqvae.quantize import VectorQuantizer2
import pytorch_lightning as pl
import omegaconf
import einops
print('✅ 所有依赖安装成功！')
"
```

### 第四步：准备数据和模型
确保你的数据和预训练模型在正确位置：
- 数据：`/kaggle/input/dataset/` 
- 预训练模型：`models/vavae-imagenet256-f16d32-dinov2.pt`

### 第五步：开始微调
```bash
!python finetune_vavae.py
```

## 📋 配置文件说明

项目包含3个独立的YAML配置文件，对应官方的3阶段训练策略：

### 阶段1：DINOv2对齐训练 (`configs/stage1_alignment.yaml`)
- **目标**：与视觉基础模型对齐
- **参数**：`vf_weight=0.5`, `disc_start=5001`, 无margin
- **训练轮数**：100 epochs

### 阶段2：重建优化训练 (`configs/stage2_reconstruction.yaml`)  
- **目标**：提升重建性能
- **参数**：`vf_weight=0.1`, `disc_start=1`, 无margin
- **训练轮数**：15 epochs

### 阶段3：Margin优化训练 (`configs/stage3_margin.yaml`)
- **目标**：进一步优化重建
- **参数**：`vf_weight=0.1`, `distmat_margin=0.25`, `cos_margin=0.5`
- **训练轮数**：15 epochs

## 🔧 自定义配置

### 修改数据路径
如果你的数据不在默认位置，请修改配置文件中的数据路径：
```yaml
data:
  params:
    train:
      target: ldm.data.imagenet.ImageNetTrain
      params:
        config:
          # 修改为你的数据路径
          size: 256
```

### 调整GPU设置
默认配置为双GPU训练。如果需要修改：
```yaml
lightning:
  trainer:
    gpus: 1  # 改为单GPU
    # strategy: ddp  # 单GPU时注释掉这行
```

### 调整批次大小
根据GPU内存调整批次大小：
```yaml
data:
  params:
    batch_size: 4  # 减小批次大小以适应GPU内存
```

## 📊 评估微调效果

训练完成后，使用评估脚本检查FID改善：
```bash
!python evaluate_vavae.py --checkpoint LightningDiT/vavae/logs/*/checkpoints/last.ckpt --test_data /kaggle/input/dataset
```

## 🐛 常见问题

### Q1: taming-transformers导入失败
**A1**: 确保使用官方安装方式（`git clone` + `pip install -e .`），而不是直接pip安装。

### Q2: 训练过程中GPU内存不足
**A2**: 减小批次大小（batch_size）或使用梯度累积：
```yaml
lightning:
  trainer:
    accumulate_grad_batches: 2
```

### Q3: checkpoint路径错误
**A3**: 训练完成后，手动检查并更新下一阶段配置文件中的`weight_init`路径。

### Q4: 数据加载失败
**A4**: 确保数据格式正确，图像文件为PNG或JPG格式，按用户目录组织。

## 📈 性能优化建议

1. **使用双GPU**：配置文件已优化为双GPU训练
2. **合理设置检查点**：每2000-5000步保存一次
3. **监控训练进度**：关注`val/rec_loss`指标
4. **分阶段训练**：严格按照3阶段策略执行

## 🎯 预期结果

- **阶段1完成**：模型与DINOv2特征对齐
- **阶段2完成**：重建质量显著提升
- **阶段3完成**：FID分数从~16降低到~5以下

## 📞 技术支持

如果遇到问题：
1. 检查依赖是否正确安装
2. 验证数据和模型路径
3. 查看训练日志中的错误信息
4. 参考官方LightningDiT项目文档

---

**注意**：本项目完全基于官方LightningDiT框架，确保与原项目的严格一致性。
