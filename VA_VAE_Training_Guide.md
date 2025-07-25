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

### **重要澄清**：
- **我们使用的模型**：`vavae-imagenet256-f16d32-dinov2.pt` (800轮，生产级)
- **HuggingFace实验版**：`vavae-imagenet256-f16d32-dinov2-50ep.ckpt` (50轮，实验级)

### **训练策略：先微调VA-VAE，再训练扩散模型**

#### 阶段1：VA-VAE微调（必需）
```yaml
# micro_doppler_vavae_config.yaml
ckpt_path: /path/to/vavae-imagenet256-f16d32-dinov2.pt  # 使用800轮版本
weight_init: /path/to/vavae-imagenet256-f16d32-dinov2.pt  # 预训练权重

model:
  base_learning_rate: 1.0e-05  # 较小学习率，微调
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    monitor: val/rec_loss
    embed_dim: 32
    use_vf: dinov2  # 保持DINOv2特征对齐
    reverse_proj: true
    lossconfig:
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        disc_start: 1000  # 延迟判别器启动
        kl_weight: 1.0e-06
        disc_weight: 0.3  # 降低判别器权重
        vf_weight: 0.05   # 降低视觉特征权重，避免过度约束
        adaptive_vf: true
    ddconfig:
      double_z: true
      z_channels: 32
      resolution: 256
      in_channels: 3    # 彩色时频图
      out_ch: 3
      ch: 128
      ch_mult: [1, 1, 2, 2, 4]
      num_res_blocks: 2
      attn_resolutions: [16]
      dropout: 0.0

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 2  # 小批次，适合小数据集
    wrap: false    # 不重复数据
    train:
      target: your.custom.MicroDopplerDataset
      params:
        data_root: /path/to/micro_doppler_data
        size: 256
        user_conditioning: true  # 启用用户条件
    validation:
      target: your.custom.MicroDopplerDataset
      params:
        data_root: /path/to/micro_doppler_val_data
        size: 256
        user_conditioning: true

lightning:
  trainer:
    devices: 1
    num_nodes: 1
    strategy: auto
    accelerator: gpu
    max_epochs: 100  # 微调轮数
    precision: 16    # 混合精度
    check_val_every_n_epoch: 5
    log_every_n_steps: 10
```

### 1. 数据准备（无数据增强）
```python
# 创建自定义数据加载器 (参考 ldm/data/imagenet.py)
class MicroDopplerDataset(Dataset):
    def __init__(self, data_root, size=256, user_conditioning=True):
        self.data_root = data_root
        self.size = size
        self.user_conditioning = user_conditioning

        # 加载31个用户的时频图文件
        self.samples = []
        for user_id in range(31):
            user_path = os.path.join(data_root, f"user_{user_id:02d}")
            if os.path.exists(user_path):
                for img_file in os.listdir(user_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(user_path, img_file),
                            'user_id': user_id
                        })

        print(f"Loaded {len(self.samples)} micro-Doppler samples from {data_root}")

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载时频图 (256x256x3)
        image = Image.open(sample['path']).convert('RGB')
        image = image.resize((self.size, self.size), Image.LANCZOS)

        # 转换为tensor并归一化到[-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        result = {'image': image}

        if self.user_conditioning:
            result['user_id'] = sample['user_id']
            result['class_label'] = sample['user_id']  # 用于条件生成

        return result

    def __len__(self):
        return len(self.samples)
```

### 2. VA-VAE微调训练命令
```bash
# 进入vavae目录
cd LightningDiT/vavae

# 启动VA-VAE微调训练
bash run_train.sh configs/micro_doppler_vavae_config.yaml
```

### 3. 监控训练过程
```python
# 关键指标监控
- val/rec_loss: 重建损失（主要指标）
- train/vf_loss: 视觉特征对齐损失
- train/disc_loss: 判别器损失
- train/kl_loss: KL散度损失

# 预期训练曲线：
# - 前10轮：快速下降（适配时频图域）
# - 10-50轮：缓慢优化（细节调整）
# - 50轮后：收敛（用户特征学习）
```

#### 阶段2：扩散模型训练（VA-VAE微调完成后）
```yaml
# lightningdit_micro_doppler_config.yaml
ckpt_path: /path/to/finetuned_vavae.ckpt  # 使用微调后的VA-VAE

model:
  model_type: LightningDiT-XL
  in_chans: 32  # VA-VAE潜在空间维度

vae:
  ckpt_path: /path/to/finetuned_vavae.ckpt  # 关键：使用微调后的VA-VAE
  downsample_ratio: 16

data:
  data_path: /path/to/micro_doppler_latents  # 预提取的潜在特征
  image_size: 256
  num_classes: 31  # 31个用户

sample:
  num_sampling_steps: 50
  cfg_scale: 4.0  # 分类器自由引导
```

### 4. 完整训练流程

#### 步骤1：VA-VAE微调（2-3天）
```bash
# 1. 准备数据集
mkdir -p /data/micro_doppler/{train,val}
# 组织为 user_00, user_01, ..., user_30 目录结构

# 2. 启动VA-VAE微调
cd LightningDiT/vavae
bash run_train.sh configs/micro_doppler_vavae_config.yaml

# 3. 验证微调效果
python ../evaluate_tokenizer.py --config configs/micro_doppler_vavae_config.yaml
```

#### 步骤2：提取潜在特征（1天）
```bash
# 使用微调后的VA-VAE提取所有训练数据的潜在特征
cd LightningDiT
python extract_features.py --config configs/micro_doppler_vavae_config.yaml
```

#### 步骤3：扩散模型训练（1-2周）
```bash
# 训练LightningDiT扩散模型
bash run_train.sh configs/lightningdit_micro_doppler_config.yaml
```

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

## 📋 微多普勒时频图训练行动计划

### **阶段1：VA-VAE微调（必需，2-3天）**
1. **准备数据集**：
   ```
   /data/micro_doppler/
   ├── train/
   │   ├── user_00/ (用户0的时频图)
   │   ├── user_01/
   │   └── ... user_30/
   └── val/
       ├── user_00/
       └── ...
   ```

2. **配置VA-VAE微调**：
   - 使用 `vavae-imagenet256-f16d32-dinov2.pt` (800轮版本)
   - 学习率：1e-5 (微调)
   - 批次大小：2 (小数据集)
   - 训练轮数：100轮

3. **启动训练**：
   ```bash
   cd LightningDiT/vavae
   bash run_train.sh configs/micro_doppler_vavae_config.yaml
   ```

### **阶段2：扩散模型训练（1-2周）**
1. **提取潜在特征**
2. **训练LightningDiT**
3. **条件生成测试**

### **关键要点**
- ✅ **不使用数据增强** (会破坏时频图特征)
- ✅ **先微调VA-VAE** (领域适配)
- ✅ **使用800轮预训练模型** (不是50轮实验版)
- ✅ **保持彩色3通道** (256×256×3)
- ✅ **用户条件生成** (31个用户ID)

这样就可以训练出专门的微多普勒时频图数据增广模型了！
