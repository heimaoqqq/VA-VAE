# 🚀 Kaggle T4*2 双GPU训练完整指南

## 🎯 核心发现

经过深入研究，发现Kaggle环境中使用双GPU的关键是：
- ❌ **不要使用** `accelerate launch` 命令
- ✅ **必须使用** `notebook_launcher` 函数
- ✅ **必须设置** Kaggle特定的环境变量

## 📋 前置要求

### 1. Kaggle设置
- 在Notebook设置中选择 **"GPU T4 x2"** 加速器
- 确保有足够的GPU配额

### 2. 数据准备
```
/kaggle/working/data_split/
├── train/
│   ├── user1/
│   ├── user2/
│   └── ...
└── val/
    ├── user1/
    ├── user2/
    └── ...
```

## 🔧 一键设置

### 步骤1: 克隆项目
```python
import os
os.chdir('/kaggle/working')

!git clone https://github.com/heimaoqqq/VA-VAE.git
os.chdir('/kaggle/working/VA-VAE')
!git submodule update --init --recursive
```

### 步骤2: 运行Kaggle专用配置
```python
!python kaggle_dual_gpu_setup.py
```

### 步骤3: 验证配置
```python
# 应该看到类似输出:
# 🎉 Kaggle T4*2 配置完成!
# ✅ notebook_launcher测试成功
```

## 🚀 训练方法

### 方法1: 完整流程 (推荐)
```python
# 一键运行完整的三阶段训练
!python kaggle_training_wrapper.py complete
```

### 方法2: 分步执行
```python
# 阶段1: 双GPU特征提取
!python kaggle_training_wrapper.py stage1

# 阶段2: 双GPU DiT训练  
!python kaggle_training_wrapper.py stage2

# 阶段3: 单GPU图像生成
!python stage3_inference.py \
    --dit_checkpoint /kaggle/working/trained_models/best_model \
    --vavae_config vavae_config.yaml \
    --output_dir /kaggle/working/generated_images \
    --user_ids 1 2 3 4 5 \
    --num_samples_per_user 4 \
    --seed 42
```

### 方法3: 在Notebook中直接调用
```python
# 导入并运行
from kaggle_training_wrapper import kaggle_complete_pipeline
kaggle_complete_pipeline()
```

## 📊 监控双GPU使用

### 实时GPU监控
```python
import subprocess
import time
from IPython.display import clear_output

def monitor_kaggle_gpu(duration_minutes=10):
    """监控Kaggle GPU使用情况"""
    end_time = time.time() + duration_minutes * 60
    
    while time.time() < end_time:
        clear_output(wait=True)
        
        # 获取GPU状态
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        print(f"🕐 {time.strftime('%H:%M:%S')} - Kaggle GPU状态")
        print("=" * 60)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 6:
                    idx, name, util, mem_used, mem_total, temp = parts[:6]
                    mem_percent = (int(mem_used) / int(mem_total)) * 100
                    print(f"GPU {idx}: {name}")
                    print(f"  🔥 使用率: {util}%")
                    print(f"  💾 内存: {mem_used}MB/{mem_total}MB ({mem_percent:.1f}%)")
                    print(f"  🌡️ 温度: {temp}°C")
                    print()
        
        print("按 Kernel -> Interrupt 停止监控")
        time.sleep(5)

# 在训练开始后运行
monitor_kaggle_gpu(duration_minutes=15)
```

## 🎯 预期输出

### 正确的双GPU配置输出:
```
🔧 Accelerator配置:
  进程数: 2
  当前进程: 0
  设备: cuda:0
  混合精度: fp16
  分布式类型: MULTI_GPU

进程 0/2
设备: cuda:0
分布式类型: DistributedType.MULTI_GPU

进程 1/2  
设备: cuda:1
分布式类型: DistributedType.MULTI_GPU
```

### GPU使用率监控:
```
GPU 0: Tesla T4
  🔥 使用率: 95%
  💾 内存: 8234MB/15360MB (53.6%)
  🌡️ 温度: 45°C

GPU 1: Tesla T4
  🔥 使用率: 94%
  💾 内存: 8156MB/15360MB (53.1%)
  🌡️ 温度: 44°C
```

## 🔍 故障排除

### 问题1: 只显示1个进程
**解决方案:**
```python
# 重启Notebook内核 (Kernel -> Restart & Clear Output)
# 重新运行配置
!python kaggle_dual_gpu_setup.py
```

### 问题2: NCCL通信错误
**解决方案:**
```python
# 已自动设置以下环境变量:
# NCCL_P2P_DISABLE=1
# NCCL_IB_DISABLE=1
# 这些设置专门针对Kaggle T4环境
```

### 问题3: 内存不足
**解决方案:**
```python
# 减少批次大小
# 在kaggle_training_wrapper.py中修改:
# batch_size = 8  # 从16改为8
```

### 问题4: 导入错误
**解决方案:**
```python
# 确保在正确目录
import os
os.chdir('/kaggle/working/VA-VAE')

# 添加路径
import sys
sys.path.append('/kaggle/working/VA-VAE')
```

## 📈 性能优化

### Kaggle T4*2 优化设置:
- **批次大小**: 每GPU 16 (总计32)
- **混合精度**: FP16
- **梯度累积**: 1步
- **学习率**: 1e-4
- **优化器**: AdamW

### 预期训练时间:
- **特征提取**: ~10分钟 (4000样本)
- **DiT训练**: ~2小时 (50轮)
- **图像生成**: ~5分钟 (40张图像)

## 🎉 成功标志

训练成功的标志:
- ✅ 两个GPU都有高使用率 (>90%)
- ✅ 内存使用均匀分布
- ✅ 训练损失正常下降
- ✅ 没有NCCL或分布式错误
- ✅ 生成的图像质量良好

## 📞 技术支持

如果遇到问题:
1. 检查Kaggle加速器设置 (必须是GPU T4 x2)
2. 确认数据路径正确
3. 重启Notebook内核
4. 重新运行配置脚本
5. 查看详细错误日志

---

**关键提醒**: Kaggle环境与本地环境不同，必须使用`notebook_launcher`而不是`accelerate launch`！

现在您可以在Kaggle上享受真正的双GPU分布式训练了！🚀
