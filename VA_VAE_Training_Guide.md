# VA-VAE训练指南：微多普勒时频图扩散模型

## 🎯 目标
基于LightningDiT的VA-VAE架构，训练适用于微多普勒时频图的扩散模型。

## 📚 VA-VAE训练说明总结

### 🔧 环境安装

1. **基础环境**：
   ```bash
   # 安装LightningDiT环境
   python install_dependencies.py
   ```

2. **VA-VAE专用依赖**：
   ```bash
   pip install -r LightningDiT/vavae/vavae_requirements.txt
   ```

3. **Taming-Transformers**（必需）：
   ```bash
   git clone https://github.com/CompVis/taming-transformers.git
   cd taming-transformers
   pip install -e .
   
   # 修复torch 2.x兼容性
   export FILE_PATH=./taming-transformers/taming/data/utils.py
   sed -i 's/from torch._six import string_classes/from six import string_types as string_classes/' "$FILE_PATH"
   ```

### 🏗️ 模型架构配置

#### 核心参数说明：

| 参数 | 含义 | 微多普勒适配建议 |
|------|------|------------------|
| `embed_dim` | 潜在空间维度 | 32 (标准) / 16 (轻量) / 64 (高质量) |
| `z_channels` | 潜在特征通道数 | 与embed_dim相同 |
| `resolution` | 输入图像分辨率 | 256 (适合时频图) |
| `in_channels` | 输入通道数 | **1 (灰度时频图) 或 3 (伪彩色)** |
| `out_ch` | 输出通道数 | **与in_channels相同** |

#### 视觉特征对齐选项：

1. **DINOv2** (`use_vf: dinov2`)：
   - 更强的语义理解
   - 适合复杂模式识别
   - **推荐用于微多普勒特征提取**

2. **MAE** (`use_vf: mae`)：
   - 更好的重建能力
   - 适合细节保持
   - 适合时频图纹理保持

3. **无特征对齐** (标准LDM)：
   - 最轻量级
   - 训练最快
   - 基础版本

## 🔬 HuggingFace预训练模型变体分析

### VA-VAE模型 (1.5GB+)
| 模型 | 配置 | 特征 | 适用场景 |
|------|------|------|----------|
| `vavae-imagenet256-f16d32-dinov2-50ep.ckpt` | f16d32 + DINOv2 | 语义对齐 | **微多普勒模式识别** |
| `vavae-imagenet256-f16d32-mae-50ep.ckpt` | f16d32 + MAE | 重建优化 | 时频图细节保持 |
| `vavae-imagenet256-f16d64-dinov2-50ep.ckpt` | f16d64 + DINOv2 | 高维语义 | 复杂微多普勒场景 |
| `vavae-imagenet256-f16d64-mae-50ep.ckpt` | f16d64 + MAE | 高维重建 | 高质量时频图生成 |

### LDM模型 (349MB)
| 模型 | 配置 | 特点 | 适用场景 |
|------|------|------|----------|
| `ldm-imagenet256-f16d16-50ep.ckpt` | f16d16 | 轻量级 | 快速原型验证 |
| `ldm-imagenet256-f16d32-50ep.ckpt` | f16d32 | 标准版 | 基础时频图生成 |
| `ldm-imagenet256-f16d64-50ep.ckpt` | f16d64 | 高质量 | 精细时频图生成 |

## 🎯 微多普勒时频图训练建议

### 1. 数据准备
```python
# 创建自定义数据加载器 (参考 ldm/data/imagenet.py)
class MicroDopplerDataset(Dataset):
    def __init__(self, data_root, size=256):
        self.data_root = data_root
        self.size = size
        # 加载时频图文件列表
        
    def __getitem__(self, idx):
        # 加载时频图 (.png/.jpg)
        # 预处理：归一化到[-1, 1]
        # 如果是灰度图，转换为3通道或修改模型配置
        pass
```

### 2. 配置文件修改
```yaml
# 基于 f16d32_vfdinov2.yaml 修改
model:
  params:
    embed_dim: 32  # 或16/64
    use_vf: dinov2  # 推荐用于微多普勒
    ddconfig:
      in_channels: 1  # 灰度时频图
      out_ch: 1       # 输出也是灰度
      resolution: 256 # 时频图分辨率

data:
  params:
    train:
      target: your.custom.MicroDopplerDataset
      params:
        data_root: /path/to/micro_doppler_data
```

### 3. 训练命令
```bash
# 从预训练模型微调
bash LightningDiT/vavae/run_train.sh your_micro_doppler_config.yaml
```

### 4. 推荐训练策略

#### 阶段1：预训练模型微调
- 使用 `vavae-imagenet256-f16d32-dinov2-50ep.ckpt` 作为起点
- 配置 `weight_init` 参数
- 较小学习率 (1e-5)

#### 阶段2：领域适配
- 逐步增加微多普勒数据比例
- 调整损失权重
- 监控重建质量

## 🔍 关键技术点

### 1. 通道数适配
- **灰度时频图**：修改 `in_channels: 1, out_ch: 1`
- **伪彩色时频图**：保持 `in_channels: 3, out_ch: 3`

### 2. 视觉特征选择
- **DINOv2**：更适合微多普勒的模式识别
- **MAE**：更适合时频图的纹理重建

### 3. 损失函数调整
- `vf_weight: 0.1`：视觉特征对齐权重
- `kl_weight: 1e-6`：KL散度权重
- `disc_weight: 0.5`：判别器权重

### 4. 训练资源
- **官方配置**：4x8 H800 GPUs
- **最小配置**：1x8 V100/A100 GPUs
- **batch_size**：根据GPU内存调整

## 📋 下一步行动计划

1. **准备微多普勒数据集**
2. **选择合适的预训练模型** (推荐DINOv2变体)
3. **修改配置文件** (通道数、数据加载器)
4. **设置训练环境** (Taming-Transformers等)
5. **开始微调训练**

这样就可以基于LightningDiT训练专门的微多普勒时频图扩散模型了！
