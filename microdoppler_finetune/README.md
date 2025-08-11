# VA-VAE 微多普勒语义对齐微调项目

## 🚀 VA-VAE 微多普勒微调 - Kaggle执行指南

基于 [LightningDiT](https://github.com/hustvl/LightningDiT) 的 Vision-Aligned VAE (VA-VAE) 完整实现。

## 📋 核心理解

### VA-VAE = AutoencoderKL + DINOv2对齐 + 反向投影

**关键组件**：
1. **DINOv2特征提取器**：提取1024维语义特征
2. **反向投影**：32维潜在空间 → 1024维DINOv2空间
3. **双重对齐损失**：
   - 距离矩阵损失（保持相对关系）
   - 余弦相似度损失（特征对齐）
4. **自适应权重平衡**：自动平衡重建和对齐损失

### 数据集特点
- **规模**：31个用户，每用户约150张图像（总计~4,650张）
- **挑战**：用户间差异细微，需要专门的对比学习
- **目标**：训练用户条件的潜在扩散模型(DiT)

## 📁 清晰的文件结构

```
microdoppler_finetune/
├── step1_setup_environment.py   # 环境配置和依赖安装
├── step2_download_models.py     # 下载预训练VA-VAE权重
├── step3_prepare_dataset.py     # 数据集划分（训练/验证）
├── step4_train_vavae.py        # 完整的三阶段VA-VAE训练
└── step5_validate_export.py     # 验证结果并导出DiT编码器
```

## ⚠️ 重要说明

1. **精度设置**：使用**32位精度**（原项目标准），避免FP16导致的梯度NaN问题
2. **无用户对比损失**：原VA-VAE仅依赖DINOv2语义对齐，不包含额外的用户对比损失
3. **GPU选择建议**：
   - **推荐P100 (16GB)**：单GPU稳定，配置简单
   - **T4×2 (32GB)**：需要特殊DDP配置，可能不稳定

## 🎯 Kaggle执行序列

### Step 1: 环境设置（2分钟）
```python
!python step1_setup_environment.py --kaggle
```
- 安装PyTorch Lightning和依赖
- 设置LightningDiT路径
- 检测GPU环境

### Step 2: 下载模型（5分钟）
```python
!python step2_download_models.py --kaggle
```
- 下载预训练VA-VAE权重
- 验证文件完整性
- 准备模型配置

### Step 3: 准备数据（3分钟）
```python
!python step3_prepare_dataset.py \
    --data_root /kaggle/input/micro-doppler-data \
    --output_dir ./
```
- 扫描数据集结构
- 创建训练/验证划分
- 保存用户标签映射

### Step 4: VA-VAE训练（10-11小时）

#### 选项A：P100单GPU（推荐）
```python
!python step4_train_vavae.py \
    --data_root /kaggle/input/micro-doppler-data \
    --pretrained_path /kaggle/input/vavae-pretrained/vavae-imagenet256-f16d32-dinov2.pt \
    --stages 1,2,3 \
    --batch_size 8 \
    --gradient_accumulation 2 \
    --kaggle
```

#### 选项B：T4×2双GPU（实验性）
```python
!python step4_train_vavae.py \
    --data_root /kaggle/input/micro-doppler-data \
    --pretrained_path /kaggle/input/vavae-pretrained/vavae-imagenet256-f16d32-dinov2.pt \
    --stages 1,2,3 \
    --batch_size 16 \
    --gradient_accumulation 1 \
    --devices "0,1" \
    --kaggle_t4 \
    --kaggle
```

#### 调试NaN问题（如遇到梯度异常）
```python
!python step4_train_vavae.py \
    --data_root /kaggle/input/micro-doppler-data \
    --pretrained_path /kaggle/input/vavae-pretrained/vavae-imagenet256-f16d32-dinov2.pt \
    --stages 1 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --detect_anomaly \
    --kaggle
```

**三阶段策略**：
- **阶段1**（30 epochs）：语义对齐，disc_start=50001，vf_weight=0.5
- **阶段2**（15 epochs）：重建优化，disc_start=1，vf_weight=0.1  
- **阶段3**（10 epochs）：边距优化，margins=0.25/0.5

### Step 5: 验证和导出（10分钟）
```python
!python step5_validate_export.py \
    --checkpoint checkpoints/stage3/last.ckpt \
    --data_root /kaggle/input/micro-doppler-data \
    --all \
    --kaggle
```
- 验证重建质量（PSNR > 25dB）
- 提取潜在空间统计
- 导出DiT训练编码器

## 💡 关键参数说明

### 训练参数
```python
--batch_size 8              # 每GPU批大小
--gradient_accumulation 2   # 梯度累积（有效批大小=16）
--num_workers 2             # 数据加载线程
--seed 42                   # 随机种子
```

### VA-VAE特定参数
```python
use_vf: 'dinov2'           # Vision Foundation模型
reverse_proj: True         # 32→1024维投影
double_z: True             # KL-VAE均值+方差
adaptive_vf: True          # 自适应VF权重
```

### 损失权重
```python
pixelloss_weight: 1.0      # 像素重建损失
kl_weight: 1e-7            # KL散度（小数据集调整）
disc_weight: 0.3-0.5       # 判别器权重
vf_weight: 0.1-0.5         # VF对齐权重（阶段依赖）
distmat_weight: 1.0        # 距离矩阵权重
cos_weight: 1.0            # 余弦相似度权重
```

## 📊 预期输出

### 训练产物
```
checkpoints/
├── stage1/
│   ├── last.ckpt
│   └── config.yaml
├── stage2/
│   ├── last.ckpt
│   └── config.yaml
├── stage3/
│   ├── last.ckpt
│   └── config.yaml
└── vavae_microdoppler_final.pt   # 最终模型
```

### 验证结果
```
reconstruction_results.png         # 重建对比图
latent_statistics.json            # 潜在空间统计
vavae_encoder_for_dit.pt         # DiT编码器
```

## 🚨 注意事项

1. **Kaggle限制**：单session最多12小时，确保在时限内完成
2. **GPU内存**：批大小8 + 梯度累积2适合P100 (16GB)
3. **checkpoint保存**：每个阶段保存2个最佳+1个最后
4. **数据路径**：确保数据集在 `/kaggle/input/micro-doppler-data`

## 🔧 故障排除

### 内存不足
```python
# 减小批大小
--batch_size 4
--gradient_accumulation 4
```

### 训练不稳定
```python
# 降低学习率
--learning_rate 5e-5
# 增强梯度裁剪
--gradient_clip_val 0.5
```

### 时间超限
```python
# 分阶段训练
--stages 1,2  # 第一个session
--stages 3    # 第二个session
```

## 🎯 下一步

训练完成后，使用微调的VA-VAE编码器训练条件DiT：
```python
python train_conditional_dit.py \
    --vae_encoder vavae_encoder_for_dit.pt \
    --data_root /kaggle/input/micro-doppler-data
```

## 📚 参考文献

- [LightningDiT](https://github.com/hustvl/LightningDiT) - 原项目实现
- [VA-VAE论文](https://arxiv.org/abs/2410.07132) - 理论基础
- [DINOv2](https://github.com/facebookresearch/dinov2) - Vision Foundation模型 自动下载（如果启用VFM对齐）

### 3. 运行VA-VAE微调

#### 方案A: 使用预训练模型微调（推荐）✅

{{ ... }}
# 使用配置文件
python vae_training/vavae_finetune.py \
    --config vae_training/vavae_microdoppler_config.yaml

# 或使用命令行参数
python vae_training/vavae_finetune.py \
    --epochs 30 \
    --lr 1e-5 \
    --freeze-layers 3 \
    --batch-size 8
```

**为什么选择微调？**
- 利用ImageNet预训练知识
- 训练时间短（2-4小时）
- 数据需求少（4,650张足够）
- 效果通常优于从头训练

#### 方案B: 从头训练（困难）⚠️

如果必须从头训练（数据分布完全不同时）：

```python
# 需要修改配置
# pretrained_path: null
# epochs: 200+
# 执行完整三阶段训练
```

**挑战：**
- 需要DINOv2模型（2GB+内存）
- 训练时间长（1-2天）
- 需要仔细调参
- 效果可能不如微调

### 4. 运行完整两阶段流程

```bash
# 方式1: 使用两阶段管理器
python models/two_stage_training.py --config configs/two_stage_finetune.yaml

# 方式2: 使用快速脚本
bash scripts/quick_start_finetune.sh both  # 两阶段
bash scripts/quick_start_finetune.sh vae   # 只VAE
bash scripts/quick_start_finetune.sh dit   # 只DiT

# 方式3: 使用主入口
python models/main_microdoppler_finetune.py --epochs 100
```

## 📊 关键参数调整建议

### VA-VAE微调参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| learning_rate | 1e-5 | 原训练的1/100 |
| freeze_layers | 3-4 | 冻结编码器前几层 |
| batch_size | 8 | 受GPU内存限制 |
| epochs | 20-30 | 通常足够 |
| distmat_weight | 0.5 | 降低VFM权重（原1.0）|
| cos_weight | 0.5 | 降低余弦权重（原1.0）|

### 条件DiT微调参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| learning_rate | 1e-4 | 条件层可稍大 |
| freeze_backbone | True | 冻结DiT主干 |
| contrastive_weight | 0.3 | 增强用户区分 |
| dropout | 0.15 | 防止过拟合 |

## 🎯 微调策略选择

### 根据数据量选择策略

| 数据量 | VA-VAE策略 | DiT策略 |
|--------|------------|---------|
| <5K（您的情况）| 冻结编码器前3-4层 | 冻结全部主干 |
| 5K-20K | 冻结前2-3层 | 冻结大部分层 |
| >20K | 可全部微调 | 可微调部分主干 |

### 根据数据相似度选择

| 与ImageNet相似度 | 建议 |
|-----------------|------|
| 高（照片类）| 直接使用预训练，最小微调 |
| 中（医学图像等）| 标准微调流程 |
| 低（时频图、雷达等）| 深度微调或考虑从头训练 |

您的微多普勒数据属于**低相似度**类别，建议：
1. VA-VAE深度微调（但不要从头训练）
2. 降低VFM损失权重
3. 增加训练轮数

## 📈 评估指标

### VA-VAE评估
- **重建损失** (MSE): < 0.01为优秀
- **LPIPS**: < 0.1为良好
- **PSNR**: > 30dB为良好
- **潜在空间可视化**: 用户聚类清晰

### DiT评估
- **FID分数**: < 50为良好
- **用户分类准确率**: > 80%为良好
- **生成多样性**: 无模式崩塌

## ⚠️ 常见问题

### 1. GPU内存不足
```python
# 解决方案
- 减小batch_size
- 使用梯度累积
- 启用mixed_precision
- 使用更小的模型（dinov2_small）
```

### 2. 训练不稳定
```python
# 解决方案
- 降低学习率
- 增加gradient_clip
- 使用warmup
- 检查数据归一化
```

### 3. 过拟合
```python
# 解决方案
- 增加dropout
- 减少可训练参数（冻结更多层）
- 使用数据增强
- 早停
```

### 4. VFM模型下载失败
```python
# 解决方案
- 手动下载DINOv2模型
- 或设置use_vfm: false（退化为普通VAE）
```

## 📝 实验记录模板

```yaml
实验名称: vavae_finetune_exp1
日期: 2024-01-10
配置:
  pretrained: vavae_f16d32.pt
  freeze_layers: 3
  learning_rate: 1e-5
  epochs: 30
  batch_size: 8
结果:
  best_val_loss: 0.0085
  reconstruction_quality: 良好
  训练时间: 3小时
观察:
  - 重建质量在15轮后趋于稳定
  - 用户特征保留良好
  - 建议下次尝试降低VFM权重
```

## 🔗 参考资源

- **LightningDiT论文**: [arxiv.org/abs/2501.01423](https://arxiv.org/abs/2501.01423)
- **GitHub仓库**: [github.com/hustvl/LightningDiT](https://github.com/hustvl/LightningDiT)
- **DINOv2**: [github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **问题反馈**: 在GitHub Issues中提问

## 📊 决策树

```
开始
 ├─ 测试预训练VA-VAE
 │   ├─ 重建良好 → 跳过VAE微调，直接DiT微调
 │   └─ 重建较差
 │       ├─ 数据与自然图像相似 → 轻度微调（10-20 epochs）
 │       └─ 数据差异很大 → 深度微调（30-50 epochs）
 └─ 完全不可用（罕见）
     └─ 考虑从头训练（200+ epochs）
```

## ✅ 总结

1. **VA-VAE通过VFM对齐实现语义感知**，这是其优势
2. **微调优于从头训练**，特别是小数据集
3. **两阶段策略**：先VAE后DiT，循序渐进
4. **关键是平衡**：保留预训练知识 vs 适应新数据

祝您实验顺利！如有问题，请参考上述故障排除或查看详细文档。
