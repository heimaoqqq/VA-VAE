# LightningDiT条件微调 - micro-Doppler数据增强

基于LightningDiT-XL-64ep预训练模型进行用户条件扩散生成，用于micro-Doppler时频图像数据增强。

## 🎯 项目目标

通过条件扩散生成指定用户的micro-Doppler时频图像，解决小样本数据增强问题，提升用户分类准确率。

## 📊 数据集配置 (已确认)

**数据集结构**:
```
/kaggle/input/dataset/
├── ID_1/          # 用户1的micro-Doppler图像
├── ID_2/          # 用户2的micro-Doppler图像  
├── ID_3/          # 用户3的micro-Doppler图像
...
└── ID_31/         # 用户31的micro-Doppler图像
```

**数据规格**:
- 👥 **用户数量**: 31个用户 (ID_1 到 ID_31)
- 🖼️ **图像格式**: 256×256 彩色 JPG 图像  
- 📁 **数据路径**: `/kaggle/input/dataset`
- 🎯 **标签映射**: ID_1→0, ID_2→1, ..., ID_31→30
- 📊 **数据划分**: 自动80%训练 / 20%验证

## 📥 模型下载状态

| 模型 | 大小 | 状态 | 描述 |
|------|------|------|------|
| VA-VAE | 2049MB | ✅ | Vision Foundation Model Aligned VAE |
| DiT-XL-64ep | 10302MB | ✅ | LightningDiT扩散模型 (微调友好) |
| latents_stats | < 1MB | ✅ | 潜在特征统计信息 |

**总计: ~12.4GB**

## 🚀 使用流程

### 第1步: 环境验证
```bash
python step3_test_inference.py
```
**作用**: 验证所有模型能正常加载，进行基础推理测试  
**预期输出**: 环境验证通过，生成测试图像

### 第2步: 数据适配测试
```bash
python step4_microdoppler_adapter.py
```
**作用**: 测试micro-Doppler数据加载和预处理  
**需要**: 修改配置文件中的数据路径

### 第3步: 条件微调训练
```bash
python step5_conditional_dit_training.py --config configs/microdoppler_finetune.yaml --data_dir /path/to/your/data
```
**作用**: 执行完整的条件微调训练  
**预期**: 生成用户特定的扩散模型

## ⚙️ 配置说明

### 关键配置文件: `configs/microdoppler_finetune.yaml`

```yaml
model:
  params:
    num_users: 10              # 🔧 需要根据实际用户数调整
    frozen_backbone: true      # 冻结主干防过拟合
    dropout: 0.15              # 高dropout防过拟合

data:
  params:
    data_dir: "/path/to/data"  # 🔧 需要修改为实际路径
    batch_size: 2              # 小batch适应显存
    val_split: 0.2

trainer:
  precision: "bf16-mixed"      # 官方推荐精度
  max_epochs: 50
  gradient_clip_val: 0.5       # 梯度裁剪
  accumulate_grad_batches: 8   # 梯度累积

optimizer:
  params:
    lr: 5.0e-6                 # 极小学习率
    weight_decay: 1.0e-3       # 强正则化
```

## 🛡️ 风险控制策略

### 过拟合防护
- ✅ **冻结主干**: 90%参数冻结，仅训练条件层
- ✅ **高Dropout**: 0.15 dropout rate
- ✅ **强正则化**: 1e-3 weight decay
- ✅ **小学习率**: 5e-6 极小学习率
- ✅ **早停机制**: 5个epoch无改善停止
- ✅ **梯度裁剪**: 0.5 gradient clipping

### 显存优化
- ✅ **混合精度**: bf16-mixed
- ✅ **小Batch**: batch_size=2
- ✅ **梯度累积**: 累积8个batch
- ✅ **检查点**: gradient checkpointing

## 📁 目录结构

```
VA-VAE/
├── models/                          # 预训练模型
│   ├── vavae-imagenet256-f16d32-dinov2.pt
│   ├── lightningdit-xl-imagenet256-64ep.pt
│   └── latents_stats.pt
├── configs/
│   └── microdoppler_finetune.yaml   # 微调配置
├── experiments/                     # 训练实验结果
├── LightningDiT/                   # 官方代码库
├── step3_test_inference.py         # 环境验证
├── step4_microdoppler_adapter.py   # 数据适配
├── step5_conditional_dit_training.py # 微调训练
└── README_CONDITIONAL_DIT.md       # 本文件
```

## 🔧 Kaggle使用说明

### 在Kaggle中的运行步骤:

1. **克隆仓库**:
   ```bash
   !git clone https://github.com/your-username/VA-VAE.git
   cd VA-VAE
   ```

2. **下载模型** (如果尚未完成):
   ```bash
   python step2_download_models.py
   ```

3. **验证环境**:
   ```bash
   python step3_test_inference.py
   ```

4. **配置数据路径**:
   编辑 `configs/microdoppler_finetune.yaml` 中的 `data_dir`

5. **开始微调**:
   ```bash
   python step5_conditional_dit_training.py \
       --config configs/microdoppler_finetune.yaml \
       --data_dir /kaggle/input/your-microdoppler-data
   ```

## 📊 预期效果

### 相比之前diffusers方案的改进:
- 🎯 **分类器识别率**: 60-70% → 85-95%
- 🎨 **用户特征保持**: 弱 → 强  
- 🚀 **训练稳定性**: 不稳定 → 稳定
- 🎭 **生成多样性**: 保持良好

### 监控指标:
- **训练损失**: 应稳定下降
- **验证损失**: 不应过早发散
- **生成图像**: 视觉质量和用户特征保持
- **分类准确率**: 下游任务性能

## ⚠️ 注意事项

### 必须修改的配置:
1. `configs/microdoppler_finetune.yaml` 中的 `data_dir`
2. `num_users` 参数 (根据实际用户数)
3. Kaggle数据集路径映射

### 训练监控:
- 观察训练/验证损失趋势
- 定期查看生成的样本图像
- 监控GPU内存使用情况
- 记录最佳检查点

### 故障排除:
- **OOM错误**: 减小batch_size或启用gradient_checkpointing
- **损失不收敛**: 调整学习率或检查数据格式
- **过拟合**: 增加dropout或减少可训练参数

## 🏆 成功标准

✅ **技术成功**:
- 训练损失稳定收敛
- 验证损失不过早发散  
- 生成图像视觉质量良好

✅ **业务成功**:
- 分类器能识别生成的用户图像
- 生成样本增强训练集效果
- 用户间差异性得到保持

## 📞 技术支持

如遇问题，请检查:
1. 模型文件是否完整下载
2. 数据路径配置是否正确
3. 依赖库版本是否兼容
4. GPU显存是否充足

---

**🎉 祝您微调成功！期待看到优秀的条件生成效果！**
