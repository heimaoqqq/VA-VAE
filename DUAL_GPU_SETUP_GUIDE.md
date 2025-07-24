# 🚀 双GPU分布式训练完整设置指南

本指南解决了Accelerate配置错误（如`torchvision::nms does not exist`）并提供完整的双GPU分布式训练环境设置。

## 📋 问题解决

### 常见错误
- `RuntimeError: operator torchvision::nms does not exist`
- `accelerate config` 命令失败
- 只有一个GPU工作，另一个GPU空闲
- 依赖包版本冲突

## 🔧 一键修复方案

### 步骤1: 克隆项目
```bash
cd /kaggle/working
git clone https://github.com/heimaoqqq/VA-VAE.git
cd VA-VAE
git submodule update --init --recursive
```

### 步骤2: 运行修复脚本
```bash
# 运行一键修复脚本
python fix_accelerate_and_setup.py
```

### 步骤3: 开始训练
```bash
# 方案1: 双GPU训练 (推荐)
./start_dual_gpu_training.sh

# 方案2: 单GPU训练 (备用)
./start_single_gpu_training.sh
```

## 📊 脚本功能

### `fix_accelerate_and_setup.py` 包含以下功能：

#### ✅ 环境检查
- 检查GPU数量和状态
- 验证CUDA可用性
- 显示GPU详细信息

#### ✅ 依赖修复
- 卸载有问题的torchvision版本
- 安装兼容的torchvision==0.16.0+cu121
- 重新安装accelerate==0.25.0
- 清理pip缓存

#### ✅ 配置创建
- 手动创建Accelerate双GPU配置
- 设置正确的分布式参数
- 启用FP16混合精度

#### ✅ 依赖安装
- 安装所有LightningDiT必需依赖
- 包括safetensors, fairscale, einops等
- 验证安装成功

#### ✅ 模型下载
- 自动下载预训练VA-VAE模型
- 保存到正确的路径
- 验证下载完整性

#### ✅ 脚本生成
- 生成双GPU训练启动脚本
- 生成单GPU备用脚本
- 设置正确的执行权限

## 🎯 生成的文件

运行修复脚本后，会生成以下文件：

### 配置文件
- `~/.cache/huggingface/accelerate/default_config.yaml` - Accelerate配置

### 训练脚本
- `start_dual_gpu_training.sh` - 双GPU训练脚本
- `start_single_gpu_training.sh` - 单GPU备用脚本

### 模型文件
- `/kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt` - 预训练模型

## 📈 预期输出

### 修复脚本成功输出：
```
🎯 修复Accelerate配置并设置双GPU分布式训练环境
======================================================================
🔄 步骤1: 检查GPU环境
✅ 检测到 2 个GPU
  GPU 0: Tesla T4
  GPU 1: Tesla T4

🔄 步骤2: 修复PyTorch和torchvision依赖
✅ 执行成功

🔄 步骤3: 创建Accelerate配置
✅ Accelerate配置已创建

🔄 步骤4: 测试Accelerate配置
✅ Accelerate工作正常
  进程数: 2
  设备: cuda:0
  分布式类型: MULTI_GPU
  混合精度: fp16

🎉 环境设置完成!
```

### 训练时的双GPU使用：
```
🎯 启动双GPU分布式训练
==========================
🔍 检查GPU状态:
index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB]
0, Tesla T4, 95 %, 8234 MiB, 15360 MiB
1, Tesla T4, 94 %, 8156 MiB, 15360 MiB

🚀 开始训练...
🔧 Accelerator配置:
  进程数: 2
  分布式类型: MULTI_GPU
  总批次大小: 32 (16×2)
```

## 🔍 故障排除

### 如果双GPU仍不工作：
1. 检查CUDA_VISIBLE_DEVICES：
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   export CUDA_VISIBLE_DEVICES=0,1
   ```

2. 验证Accelerate配置：
   ```bash
   accelerate env
   ```

3. 使用单GPU备用方案：
   ```bash
   ./start_single_gpu_training.sh
   ```

### 如果依赖安装失败：
1. 清理环境：
   ```bash
   pip cache purge
   pip uninstall torch torchvision accelerate -y
   ```

2. 重新运行修复脚本：
   ```bash
   python fix_accelerate_and_setup.py
   ```

## 📊 监控训练

### 实时监控GPU使用：
```bash
# 基础监控
watch -n 1 nvidia-smi

# 详细监控
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv'
```

### 查看训练日志：
```bash
# 查看输出目录
ls /kaggle/working/outputs/

# 查看TensorBoard日志
tensorboard --logdir /kaggle/working/outputs/trained_models/
```

## 🎉 成功标志

训练成功启动的标志：
- ✅ 两个GPU都有高使用率（>90%）
- ✅ 内存使用均匀分布
- ✅ 训练损失正常下降
- ✅ 没有NCCL或分布式错误

## 📞 支持

如果遇到问题：
1. 检查错误日志
2. 验证数据路径是否正确
3. 确认所有依赖已安装
4. 使用单GPU备用方案测试

现在您可以享受真正的双GPU分布式训练了！🚀
