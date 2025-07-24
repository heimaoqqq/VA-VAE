#!/usr/bin/env python3
"""
修复Accelerate配置并设置双GPU分布式训练环境
解决torchvision::nms错误和accelerate config失败问题
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import torch

def print_step(step, description):
    """打印步骤信息"""
    print(f"\n{'='*60}")
    print(f"🔄 步骤{step}: {description}")
    print('='*60)

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

def check_gpu_environment():
    """检查GPU环境"""
    print_step(1, "检查GPU环境")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")
    
    return gpu_count >= 2

def fix_torch_dependencies():
    """修复PyTorch相关依赖"""
    print_step(2, "修复PyTorch和torchvision依赖")
    
    # 卸载可能有问题的包
    packages_to_remove = ["torchvision", "accelerate"]
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"卸载 {package}", ignore_error=True)
    
    # 重新安装兼容版本
    print("\n📦 重新安装兼容版本...")
    
    # 安装torchvision
    success, _ = run_command(
        "pip install torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121",
        "安装兼容的torchvision"
    )
    
    if not success:
        print("⚠️  尝试安装默认版本的torchvision")
        run_command("pip install torchvision", "安装默认torchvision")
    
    # 安装accelerate
    run_command("pip install accelerate==0.25.0", "安装Accelerate")
    
    # 清理缓存
    run_command("pip cache purge", "清理pip缓存", ignore_error=True)
    
    return True

def create_accelerate_config():
    """手动创建Accelerate配置文件"""
    print_step(3, "创建Accelerate配置")
    
    # 创建配置目录
    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 双GPU配置
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
    
    print(f"✅ Accelerate配置已创建: {config_file}")
    return config_file

def test_accelerate():
    """测试Accelerate配置"""
    print_step(4, "测试Accelerate配置")
    
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        
        print("✅ Accelerate工作正常")
        print(f"  进程数: {accelerator.num_processes}")
        print(f"  设备: {accelerator.device}")
        print(f"  分布式类型: {accelerator.distributed_type}")
        print(f"  混合精度: {accelerator.mixed_precision}")
        
        return True
    except Exception as e:
        print(f"❌ Accelerate测试失败: {e}")
        return False

def install_additional_dependencies():
    """安装额外的必需依赖"""
    print_step(5, "安装LightningDiT依赖")
    
    # 必需的依赖包
    required_packages = [
        "safetensors>=0.3.0",
        "fairscale>=0.4.13",
        "einops>=0.6.0",
        "timm>=0.9.0",
        "torchdiffeq>=0.2.3",
        "omegaconf>=2.3.0",
        "diffusers>=0.20.0",
        "pytorch-fid>=0.3.0",
        "scipy>=1.9.0"
    ]
    
    success_count = 0
    for package in required_packages:
        success, _ = run_command(f"pip install {package}", f"安装 {package}")
        if success:
            success_count += 1
    
    print(f"✅ 成功安装 {success_count}/{len(required_packages)} 个依赖包")
    return success_count == len(required_packages)

def download_pretrained_model():
    """下载预训练模型"""
    print_step(6, "下载预训练VA-VAE模型")
    
    pretrained_dir = Path("/kaggle/working/pretrained")
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    model_url = "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt"
    model_path = pretrained_dir / "vavae-imagenet256-f16d32-dinov2.pt"
    
    if model_path.exists():
        print(f"✅ 模型已存在: {model_path}")
        return True
    
    success, _ = run_command(f"wget -O {model_path} {model_url}", "下载VA-VAE模型")
    
    if success and model_path.exists():
        print(f"✅ 模型下载成功: {model_path}")
        return True
    else:
        print("❌ 模型下载失败")
        return False

def create_training_script():
    """创建训练启动脚本"""
    print_step(7, "创建训练启动脚本")
    
    script_content = '''#!/bin/bash
# 双GPU分布式训练启动脚本

echo "🎯 启动双GPU分布式训练"
echo "=========================="

# 检查GPU
echo "🔍 检查GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

# 运行训练
echo "🚀 开始训练..."
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \\
    run_complete_pipeline.py \\
    --data_dir /kaggle/working/data_split \\
    --output_dir /kaggle/working/outputs \\
    --vavae_config vavae_config.yaml \\
    --batch_size 16 \\
    --max_epochs 50 \\
    --lr 1e-4 \\
    --seed 42 \\
    --user_ids 1 2 3 4 5 \\
    --num_samples_per_user 4

echo "✅ 训练完成!"
'''
    
    script_path = Path("start_dual_gpu_training.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(script_path, 0o755)
    
    print(f"✅ 训练脚本已创建: {script_path}")
    return script_path

def create_single_gpu_fallback():
    """创建单GPU备用脚本"""
    print_step(8, "创建单GPU备用脚本")
    
    script_content = '''#!/bin/bash
# 单GPU训练备用脚本 (如果双GPU有问题)

echo "🎯 启动单GPU训练 (备用方案)"
echo "============================="

# 检查GPU
echo "🔍 检查GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行训练
echo "🚀 开始单GPU训练..."
python run_complete_pipeline.py \\
    --data_dir /kaggle/working/data_split \\
    --output_dir /kaggle/working/outputs_single_gpu \\
    --vavae_config vavae_config.yaml \\
    --batch_size 32 \\
    --max_epochs 50 \\
    --lr 1e-4 \\
    --seed 42 \\
    --user_ids 1 2 3 4 5 \\
    --num_samples_per_user 4

echo "✅ 单GPU训练完成!"
'''
    
    script_path = Path("start_single_gpu_training.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(script_path, 0o755)
    
    print(f"✅ 单GPU备用脚本已创建: {script_path}")
    return script_path

def main():
    """主函数"""
    print("🎯 修复Accelerate配置并设置双GPU分布式训练环境")
    print("=" * 70)
    print("解决torchvision::nms错误和accelerate config失败问题")
    print("=" * 70)
    
    # 检查GPU环境
    if not check_gpu_environment():
        print("❌ GPU环境检查失败")
        return False
    
    # 修复依赖
    if not fix_torch_dependencies():
        print("❌ 依赖修复失败")
        return False
    
    # 创建配置
    config_file = create_accelerate_config()
    
    # 测试配置
    if not test_accelerate():
        print("❌ Accelerate配置测试失败")
        return False
    
    # 安装额外依赖
    install_additional_dependencies()
    
    # 下载模型
    download_pretrained_model()
    
    # 创建训练脚本
    dual_gpu_script = create_training_script()
    single_gpu_script = create_single_gpu_fallback()
    
    # 完成总结
    print("\n" + "="*70)
    print("🎉 环境设置完成!")
    print("="*70)
    print(f"📁 Accelerate配置: {config_file}")
    print(f"🚀 双GPU训练脚本: {dual_gpu_script}")
    print(f"🔄 单GPU备用脚本: {single_gpu_script}")
    
    print("\n🎯 使用方法:")
    print("1. 双GPU训练 (推荐):")
    print("   ./start_dual_gpu_training.sh")
    print("\n2. 单GPU训练 (备用):")
    print("   ./start_single_gpu_training.sh")
    print("\n3. 手动运行:")
    print("   accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml run_complete_pipeline.py ...")
    
    print("\n📊 监控GPU使用:")
    print("   watch -n 1 nvidia-smi")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 所有设置完成，可以开始训练了!")
    else:
        print("\n❌ 设置过程中出现错误，请检查日志")
        sys.exit(1)
