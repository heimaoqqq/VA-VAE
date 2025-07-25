#!/usr/bin/env python3
"""
Kaggle T4*2 双GPU正确配置脚本
基于Kaggle环境的特殊要求，使用notebook_launcher而不是accelerate launch
"""

import os
import sys
import subprocess
import shutil
import torch
from pathlib import Path

def print_step(step, description):
    """打印步骤信息"""
    print(f"\n{'='*70}")
    print(f"🔄 步骤{step}: {description}")
    print('='*70)

def run_command(cmd, description="", ignore_error=False):
    """运行命令并处理错误"""
    print(f"\n🔧 {description}")
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ 执行成功")
        if result.stdout.strip():
            print(f"输出: {result.stdout.strip()}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if ignore_error:
            print(f"⚠️  命令失败但忽略: {e}")
            return False, e.stderr
        else:
            print(f"❌ 执行失败: {e}")
            if e.stderr:
                print(f"错误: {e.stderr}")
            return False, e.stderr

def check_kaggle_environment():
    """检查Kaggle环境"""
    print_step(1, "检查Kaggle环境")
    
    # 检查是否在Kaggle环境
    kaggle_indicators = [
        '/kaggle/working' in os.getcwd(),
        os.path.exists('/kaggle/input'),
        'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    ]
    
    if not any(kaggle_indicators):
        print("⚠️  警告: 似乎不在Kaggle环境中")
    else:
        print("✅ 确认在Kaggle环境中")
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    if gpu_count < 2:
        print("❌ 需要至少2个GPU，请在Kaggle设置中选择 'GPU T4 x2'")
        return False
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return True

def install_kaggle_optimized_packages():
    """安装Kaggle优化的包"""
    print_step(2, "安装Kaggle优化的依赖包")
    
    # Kaggle环境特定的包版本
    packages = [
        "accelerate==0.25.0",
        "safetensors>=0.4.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0"
    ]
    
    success_count = 0
    for package in packages:
        success, _ = run_command(f"pip install {package} --quiet", f"安装 {package}")
        if success:
            success_count += 1
    
    print(f"✅ 成功安装 {success_count}/{len(packages)} 个包")
    return success_count == len(packages)

def create_kaggle_accelerate_config():
    """创建Kaggle专用的Accelerate配置"""
    print_step(3, "创建Kaggle专用Accelerate配置")
    
    # 清理旧配置
    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    if config_dir.exists():
        shutil.rmtree(config_dir)
        print("🗑️ 已清理旧配置")
    
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Kaggle T4*2 专用配置
    config_content = """compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    config_file = config_dir / "default_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Kaggle配置已创建: {config_file}")
    
    # 设置Kaggle特定的环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Kaggle T4特定设置
    os.environ['NCCL_IB_DISABLE'] = '1'   # 禁用InfiniBand
    
    print("✅ Kaggle环境变量已设置")
    return config_file

def test_notebook_launcher():
    """测试notebook_launcher功能"""
    print_step(4, "验证双GPU环境")

    # 检查基础GPU环境
    import torch
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"✅ GPU数量: {torch.cuda.device_count()}")

    # 检查Accelerate配置
    try:
        from accelerate import Accelerator
        # 不在这里初始化Accelerator，避免CUDA初始化
        print("✅ Accelerate库可用")
    except Exception as e:
        print(f"❌ Accelerate导入失败: {e}")
        return False

    # 检查notebook_launcher
    try:
        from accelerate import notebook_launcher
        print("✅ notebook_launcher可用")
    except Exception as e:
        print(f"❌ notebook_launcher导入失败: {e}")
        return False

    print("⚠️  跳过实际的notebook_launcher测试以避免CUDA初始化冲突")
    print("✅ 环境验证完成，notebook_launcher应该可以正常工作")

    return True

def create_kaggle_training_wrapper():
    """创建Kaggle训练包装器"""
    print_step(5, "创建Kaggle训练包装器")
    
    # 创建使用notebook_launcher的训练包装器
    wrapper_code = '''#!/usr/bin/env python3
"""
Kaggle双GPU训练包装器
使用notebook_launcher而不是accelerate launch
"""

import os
import sys
from accelerate import notebook_launcher

def kaggle_stage1_extract_features():
    """阶段1: 特征提取 (Kaggle双GPU版本)"""

    def extract_features_worker():
        # 安装缺失的依赖
        import subprocess
        import sys

        # 安装torchdiffeq和其他可能缺失的依赖
        required_packages = [
            "torchdiffeq>=0.2.3",
            "scipy>=1.9.0",
            "einops>=0.6.0",
            "omegaconf>=2.3.0"
        ]

        for package in required_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"],
                             check=True, capture_output=True)
                print(f"✅ 已安装: {package}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  安装失败: {package} - {e}")

        # 导入必要的模块
        sys.path.append('/kaggle/working/VA-VAE')

        # 设置参数
        import argparse
        from stage1_extract_features import main
        
        # 模拟命令行参数
        class Args:
            data_dir = '/kaggle/working/data_split'
            vavae_config = 'vavae_config.yaml'
            output_path = '/kaggle/working/latent_features'
            batch_size = 16
            seed = 42
        
        args = Args()
        
        # 运行特征提取
        main(args)
    
    # 使用notebook_launcher启动双GPU训练
    print("🚀 启动双GPU特征提取...")
    notebook_launcher(extract_features_worker, num_processes=2)
    print("✅ 特征提取完成")

def kaggle_stage2_train_dit():
    """阶段2: DiT训练 (Kaggle双GPU版本)"""

    def train_dit_worker():
        # 安装缺失的依赖
        import subprocess
        import sys

        # 安装torchdiffeq和其他可能缺失的依赖
        required_packages = [
            "torchdiffeq>=0.2.3",
            "scipy>=1.9.0",
            "einops>=0.6.0",
            "omegaconf>=2.3.0"
        ]

        for package in required_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"],
                             check=True, capture_output=True)
                print(f"✅ 已安装: {package}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  安装失败: {package} - {e}")

        # 导入必要的模块
        sys.path.append('/kaggle/working/VA-VAE')

        # 设置参数
        from stage2_train_dit import main
        
        # 模拟命令行参数
        class Args:
            latent_dir = '/kaggle/working/latent_features'
            output_dir = '/kaggle/working/trained_models'
            model_name = 'LightningDiT-XL/1'
            batch_size = 16
            max_epochs = 50
            lr = 1e-4
            seed = 42
            save_every = 10
        
        args = Args()
        
        # 运行DiT训练
        main(args)
    
    # 使用notebook_launcher启动双GPU训练
    print("🚀 启动双GPU DiT训练...")
    notebook_launcher(train_dit_worker, num_processes=2)
    print("✅ DiT训练完成")

def kaggle_complete_pipeline():
    """完整的Kaggle双GPU训练流程"""
    print("🎯 Kaggle双GPU完整训练流程")
    print("=" * 50)
    
    # 阶段1: 特征提取
    print("\\n📊 阶段1: 特征提取")
    kaggle_stage1_extract_features()
    
    # 阶段2: DiT训练
    print("\\n🤖 阶段2: DiT训练")
    kaggle_stage2_train_dit()
    
    # 阶段3: 图像生成 (单GPU即可)
    print("\\n🎨 阶段3: 图像生成")
    os.system("""python stage3_inference.py \\
        --dit_checkpoint /kaggle/working/trained_models/best_model \\
        --vavae_config vavae_config.yaml \\
        --output_dir /kaggle/working/generated_images \\
        --user_ids 1 2 3 4 5 \\
        --num_samples_per_user 4 \\
        --seed 42""")
    
    print("🎉 完整流程完成!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "stage1":
            kaggle_stage1_extract_features()
        elif sys.argv[1] == "stage2":
            kaggle_stage2_train_dit()
        elif sys.argv[1] == "complete":
            kaggle_complete_pipeline()
        else:
            print("用法: python kaggle_training_wrapper.py [stage1|stage2|complete]")
    else:
        kaggle_complete_pipeline()
'''
    
    wrapper_file = Path("kaggle_training_wrapper.py")
    with open(wrapper_file, 'w') as f:
        f.write(wrapper_code)
    
    os.chmod(wrapper_file, 0o755)
    print(f"✅ Kaggle训练包装器已创建: {wrapper_file}")
    
    return wrapper_file

def create_kaggle_usage_guide():
    """创建Kaggle使用指南"""
    print_step(6, "创建Kaggle使用指南")
    
    guide_content = '''# 🚀 Kaggle T4*2 双GPU训练指南

## 📋 前置要求

1. **Kaggle设置**: 确保选择了 "GPU T4 x2" 加速器
2. **数据准备**: 数据应该在 `/kaggle/working/data_split/` 目录下

## 🔧 环境设置

```python
# 1. 克隆项目
import os
os.chdir('/kaggle/working')
!git clone https://github.com/heimaoqqq/VA-VAE.git
os.chdir('/kaggle/working/VA-VAE')
!git submodule update --init --recursive

# 2. 运行Kaggle专用配置
!python kaggle_dual_gpu_setup.py

# 3. 验证配置
from accelerate import notebook_launcher
print("✅ notebook_launcher可用")
```

## 🚀 训练方法

### 方法1: 完整流程 (推荐)
```python
!python kaggle_training_wrapper.py complete
```

### 方法2: 分步执行
```python
# 阶段1: 特征提取
!python kaggle_training_wrapper.py stage1

# 阶段2: DiT训练
!python kaggle_training_wrapper.py stage2

# 阶段3: 图像生成
!python stage3_inference.py --dit_checkpoint /kaggle/working/trained_models/best_model --vavae_config vavae_config.yaml --output_dir /kaggle/working/generated_images --user_ids 1 2 3 4 5 --num_samples_per_user 4 --seed 42
```

### 方法3: 在Notebook中直接使用
```python
from kaggle_training_wrapper import kaggle_complete_pipeline
kaggle_complete_pipeline()
```

## 📊 监控GPU使用

```python
import subprocess
import time
from IPython.display import clear_output

def monitor_kaggle_gpu():
    for i in range(20):  # 监控20次
        clear_output(wait=True)
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("🔍 Kaggle GPU状态:")
        print(result.stdout)
        time.sleep(5)

# 在训练开始后运行
monitor_kaggle_gpu()
```

## 🎯 预期输出

正确配置后应该看到:
```
进程 0/2
设备: cuda:0
分布式类型: DistributedType.MULTI_GPU

进程 1/2
设备: cuda:1
分布式类型: DistributedType.MULTI_GPU
```

## 🔍 故障排除

1. **如果只显示1个进程**: 重启Notebook内核，重新运行配置
2. **如果NCCL错误**: 已设置 `NCCL_P2P_DISABLE=1`
3. **如果内存不足**: 减少batch_size到8或12

## 📞 支持

- 确保选择了正确的Kaggle加速器 (GPU T4 x2)
- 确保数据路径正确
- 如有问题，检查Kaggle系统日志
'''
    
    guide_file = Path("KAGGLE_DUAL_GPU_GUIDE.md")
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Kaggle使用指南已创建: {guide_file}")
    return guide_file

def main():
    """主函数"""
    print("🎯 Kaggle T4*2 双GPU正确配置脚本")
    print("=" * 70)
    print("基于notebook_launcher的Kaggle专用配置")
    print("=" * 70)
    
    # 检查环境
    if not check_kaggle_environment():
        print("❌ 环境检查失败")
        return False
    
    # 安装包
    if not install_kaggle_optimized_packages():
        print("❌ 包安装失败")
        return False
    
    # 创建配置
    config_file = create_kaggle_accelerate_config()
    
    # 测试功能
    if not test_notebook_launcher():
        print("❌ notebook_launcher测试失败")
        return False
    
    # 创建训练包装器
    wrapper_file = create_kaggle_training_wrapper()
    
    # 创建使用指南
    guide_file = create_kaggle_usage_guide()
    
    # 完成总结
    print("\n" + "="*70)
    print("🎉 Kaggle T4*2 配置完成!")
    print("="*70)
    print(f"📁 配置文件: {config_file}")
    print(f"🚀 训练包装器: {wrapper_file}")
    print(f"📖 使用指南: {guide_file}")
    
    print("\n🎯 使用方法:")
    print("1. 完整训练:")
    print("   python kaggle_training_wrapper.py complete")
    print("\n2. 分步训练:")
    print("   python kaggle_training_wrapper.py stage1")
    print("   python kaggle_training_wrapper.py stage2")
    print("\n3. Notebook中使用:")
    print("   from kaggle_training_wrapper import kaggle_complete_pipeline")
    print("   kaggle_complete_pipeline()")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Kaggle双GPU环境配置成功!")
        print("现在可以开始双GPU训练了! 🚀")
    else:
        print("\n❌ 配置过程中出现错误")
        sys.exit(1)
