#!/usr/bin/env python3
"""
步骤1: VA-VAE微调环境配置
配置微多普勒数据微调所需的环境
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_taming_transformers(base_path):
    """设置taming-transformers（克隆方式）"""
    print("\n📥 设置taming-transformers...")
    
    taming_dir = base_path / "taming-transformers"
    if not taming_dir.exists():
        print("📥 克隆taming-transformers...")
        try:
            subprocess.check_call([
                "git", "clone", 
                "https://github.com/CompVis/taming-transformers.git",
                str(taming_dir)
            ])
            print("✅ taming-transformers 克隆成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 克隆失败: {e}")
            return False
    else:
        print("✅ taming-transformers已存在")
    
    # 修复torch兼容性问题
    utils_file = taming_dir / "taming" / "data" / "utils.py"
    if utils_file.exists():
        print("🔧 修复torch兼容性...")
        try:
            content = utils_file.read_text()
            if "from torch._six import string_classes" in content:
                content = content.replace(
                    "from torch._six import string_classes",
                    "from six import string_types as string_classes"
                )
                utils_file.write_text(content)
                print("✅ 兼容性修复完成")
        except Exception as e:
            print(f"⚠️ 兼容性修复失败: {e}")
    
    # 添加到Python路径
    taming_path = str(taming_dir.absolute())
    if taming_path not in sys.path:
        sys.path.insert(0, taming_path)
        print(f"📂 已添加taming路径: {taming_path}")
    
    # 保存路径信息
    path_file = base_path / ".taming_path"
    with open(path_file, "w") as f:
        f.write(taming_path)
    
    return True

def setup_vavae_environment():
    """配置VA-VAE微调环境"""
    print("🔧 配置VA-VAE微调环境")
    print("="*60)
    
    # 检测运行环境
    if os.path.exists('/kaggle/working'):
        print("📍 检测到Kaggle环境")
        base_path = Path('/kaggle/working')
    else:
        print("📍 检测到本地环境")
        base_path = Path.cwd()
    
    # 创建必要的目录结构
    dirs_to_create = [
        'models',           # 存放预训练模型
        'checkpoints',      # 存放训练检查点
        'logs',            # 训练日志
        'configs',         # 配置文件
        'data_split',      # 数据集划分信息
        'visualizations'   # 可视化结果
    ]
    
    for dir_name in dirs_to_create:
        dir_path = base_path / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")
    
    print("\n📦 安装必要的Python包...")
    
    # 基础依赖
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",  # 用于DINOv2
        "einops>=0.6.0",
        "omegaconf>=2.3.0",
        "Pillow>=9.5.0",
        "numpy<2.0",  # 避免版本冲突
        "pandas",
        "matplotlib",
        "seaborn",
        "tensorboard",
        "tqdm",
        # LightningDiT/VA-VAE特定依赖
        "diffusers>=0.20.0",
        "accelerate>=0.20.0",
        "lpips>=0.1.4",  # 感知损失
        "timm>=0.9.0",   # Vision Transformer模型
        # LightningDiT transport模块依赖
        "torchdiffeq",   # ODE求解器
        "torchsde",      # SDE求解器
        "scipy",         # 科学计算
        # LightningDiT模型依赖
        "fairscale",     # 模型并行化
        "einops",        # 张量操作
        "safetensors",   # 安全模型存储
        "pytorch_fid",   # FID计算
        "tensorboard",   # 训练监控
        "omegaconf",     # 配置管理
        # 注意：不要包含clip-by-openai，有依赖冲突
        # 注意：不要包含taming，使用克隆方式
    ]
    
    # 安装基础包
    for package in packages:
        try:
            # 只跳过真正的torch核心包，不跳过torchdiffeq等
            torch_core_packages = ['torch>=', 'torchvision>=', 'pytorch-lightning>=']
            if any(package.startswith(core) for core in torch_core_packages) and os.path.exists('/kaggle/working'):
                # Kaggle已预装PyTorch核心包
                print(f"⏭️ 跳过 {package} (Kaggle预装)")
                continue
            
            print(f"📥 安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 安装 {package} 失败: {e}")
            print("   尝试继续...")
    
    # 设置taming-transformers（使用克隆方式）
    setup_taming_transformers(base_path)
    
    # 克隆或链接LightningDiT代码
    print("\n📂 配置LightningDiT代码库...")
    
    if os.path.exists('/kaggle/working'):
        # Kaggle环境：假设代码已上传
        lightningdit_path = base_path / 'LightningDiT'
        if not lightningdit_path.exists():
            print("⚠️ 请确保已上传LightningDiT代码到Kaggle")
            print("   或使用git clone: https://github.com/hustvl/LightningDiT.git")
    else:
        # 本地环境：检查父目录
        parent_lightningdit = base_path.parent / 'LightningDiT'
        if parent_lightningdit.exists():
            print(f"✅ 找到LightningDiT: {parent_lightningdit}")
            # 添加到Python路径
            sys.path.insert(0, str(parent_lightningdit / 'vavae'))
            sys.path.insert(0, str(parent_lightningdit))
        else:
            print("⚠️ 未找到LightningDiT，需要克隆代码库")
            clone_cmd = "git clone https://github.com/hustvl/LightningDiT.git"
            print(f"   运行: {clone_cmd}")
    
    # 检查GPU
    print("\n🖥️ 检查GPU配置...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ 检测到 {gpu_count} 个GPU")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")
    except ImportError:
        print("❌ PyTorch未正确安装")
    
    # 保存环境信息
    env_info = {
        'base_path': str(base_path),
        'python_version': sys.version,
        'platform': sys.platform,
        'cuda_available': torch.cuda.is_available() if 'torch' in sys.modules else False
    }
    
    import json
    with open(base_path / 'environment_info.json', 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print("\n✅ 环境配置完成!")
    print("\n下一步:")
    print("1. 运行 python step2_download_models.py 下载预训练模型")
    print("2. 运行 python step3_prepare_dataset.py 准备数据集")
    print("3. 运行 python step4_train_stage1.py 开始第一阶段训练")
    
    return base_path

if __name__ == "__main__":
    setup_vavae_environment()
