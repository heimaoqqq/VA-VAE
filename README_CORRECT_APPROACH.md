# 微多普勒用户条件化生成 - 正确方法

基于LightningDiT原项目的正确实现方法

## 🎯 项目理解

### 核心概念
- **VA-VAE**: 视觉基础模型对齐的变分自编码器，负责图像↔潜在空间转换
- **DiT**: 扩散Transformer，在潜在空间进行扩散去噪
- **用户条件化**: 在DiT中添加用户ID作为类别条件

### 与原方法的区别

| 方面 | ❌ 错误方法 (之前) | ✅ 正确方法 (现在) |
|------|------------------|------------------|
| **VAE使用** | 实时编码解码，参与训练 | 预提取特征，不参与训练 |
| **训练目标** | VAE重建损失 | 扩散去噪损失 |
| **数据格式** | 原始图像 (3, 256, 256) | 潜在特征 (32, 16, 16) |
| **用户条件** | 在VAE编码器中添加 | 在DiT中作为类别条件 |
| **内存效率** | 低 (需要存储大图像) | 高 (只需存储小特征) |
| **训练稳定性** | 不稳定 (KL散度爆炸) | 稳定 (成熟的扩散训练) |

## 🚀 快速开始

### 环境准备
```bash
# 确保已安装依赖
pip install torch torchvision pytorch-lightning
pip install safetensors transformers accelerate
pip install matplotlib pillow tqdm
```

### 一键运行完整流程
```bash
python run_complete_pipeline.py \
    --data_dir /kaggle/working/data_split \
    --vavae_path /kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt \
    --output_dir /kaggle/working/correct_outputs \
    --batch_size 32 \
    --max_epochs 50 \
    --devices 2 \
    --generate_user_ids 1 2 3 4 5 \
    --num_samples_per_user 4
```

## 📋 分阶段执行

### 阶段1: 特征提取
```bash
python stage1_extract_features.py \
    --data_dir /kaggle/working/data_split \
    --vavae_path /kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt \
    --output_path /kaggle/working/latent_features \
    --batch_size 32
```

**输出**: 
- `train.safetensors`: 训练集潜在特征 (N, 32, 16, 16)
- `val.safetensors`: 验证集潜在特征 (M, 32, 16, 16)

### 阶段2: DiT训练
```bash
python stage2_train_dit.py \
    --latent_dir /kaggle/working/latent_features \
    --output_dir /kaggle/working/trained_models \
    --batch_size 32 \
    --max_epochs 100 \
    --lr 1e-4 \
    --devices 2 \
    --precision 16-mixed
```

**输出**:
- `checkpoints/`: 模型检查点
- `lightning_logs/`: 训练日志

### 阶段3: 图像生成
```bash
python stage3_inference.py \
    --dit_checkpoint /kaggle/working/trained_models/checkpoints/best.ckpt \
    --vavae_path /kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt \
    --output_dir /kaggle/working/generated_images \
    --user_ids 1 2 3 4 5 \
    --num_samples_per_user 4 \
    --guidance_scale 4.0 \
    --num_steps 250
```

**输出**:
- 单独的生成图像: `micro_doppler_user01_001.png`
- 网格图像: `micro_doppler_grid.png`

## 🔧 技术细节

### 数据流程
```
原始图像 (3, 256, 256)
    ↓ VA-VAE.encode()
潜在特征 (32, 16, 16)
    ↓ DiT + 扩散训练
训练好的DiT模型
    ↓ DiT.sample() + 用户条件
新的潜在特征 (32, 16, 16)
    ↓ VA-VAE.decode()
生成图像 (3, 256, 256)
```

### 模型架构
```python
UserConditionedDiT:
├── DiT Backbone (LightningDiT)
│   ├── Patch Embedding: (32, 16, 16) → patches
│   ├── Transformer Blocks: 28层
│   ├── User Condition: 嵌入用户ID
│   └── Output: 预测噪声/速度
└── Transport: 扩散采样器
```

### 关键参数
- **潜在空间**: 32通道, 16×16分辨率 (f16d32)
- **下采样比例**: 16倍 (256→16)
- **用户条件**: 类别条件，支持classifier-free guidance
- **扩散预测**: 速度预测 (velocity prediction)
- **采样器**: Linear transport

## 📊 预期结果

### 训练指标
- **训练损失**: 应该稳定下降，收敛到0.01-0.1
- **验证损失**: 与训练损失接近，无明显过拟合
- **训练时间**: 约2-3小时/epoch (双GPU)

### 生成质量
- **用户特异性**: 不同用户生成的图像应有明显差异
- **图像质量**: 清晰的时频结构，无明显伪影
- **多样性**: 同一用户的多个样本应有合理变化

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   --batch_size 16
   
   # 使用梯度检查点
   # 在DiT模型中启用checkpointing
   ```

2. **训练损失不收敛**
   ```bash
   # 调整学习率
   --lr 5e-5
   
   # 增加训练轮数
   --max_epochs 200
   ```

3. **生成图像质量差**
   ```bash
   # 增加采样步数
   --num_steps 500
   
   # 调整引导强度
   --guidance_scale 6.0
   ```

## 📈 性能优化

### 训练优化
- 使用混合精度: `--precision 16-mixed`
- 多GPU训练: `--devices 2`
- 数据并行: 自动启用DDP

### 推理优化
- 批量生成: 一次生成多个样本
- 缓存模型: 避免重复加载
- GPU推理: 使用CUDA加速

## 🎯 下一步

1. **评估生成质量**: 计算FID、IS等指标
2. **用户研究**: 邀请用户评估生成的微多普勒图像
3. **模型优化**: 尝试不同的DiT架构和超参数
4. **应用扩展**: 集成到实际的雷达系统中

## 📚 参考资料

- [LightningDiT原项目](https://github.com/hustvl/LightningDiT)
- [DiT论文](https://arxiv.org/abs/2212.09748)
- [VA-VAE相关工作](https://arxiv.org/abs/2112.10752)
- [扩散模型综述](https://arxiv.org/abs/2006.11239)
