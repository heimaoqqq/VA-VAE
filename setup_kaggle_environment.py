#!/usr/bin/env python3
"""
Kaggle环境依赖安装脚本
专门为增强条件扩散系统准备依赖
"""

import subprocess
import sys
import pkg_resources
from pathlib import Path

def install_package(package, description=""):
    """安装Python包"""
    print(f"🔧 安装 {package}...")
    if description:
        print(f"   {description}")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 安装失败: {e}")
        return False

def check_package(package_name):
    """检查包是否已安装"""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def main():
    print("🚀 初始化Kaggle环境：LightningDiT-S扩散模型训练")
    print("=" * 50)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    # 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"🔥 CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("⚠️ PyTorch未检测到")
    
    print("\n📦 安装必需依赖...")
    
    # 核心依赖列表
    dependencies = [
        # 深度学习框架 (Kaggle通常预装，但可能需要更新)
        ("torch>=2.0.0", "PyTorch深度学习框架"),
        ("torchvision", "PyTorch视觉工具"),
        
        # 分布式训练
        ("accelerate", "Hugging Face Accelerate分布式训练"),
        
        # 扩散模型 (DiT-S需要)
        ("diffusers==0.32.1", "Hugging Face Diffusers库"),
        ("torchdiffeq", "ODE求解器，LightningDiT transport需要"),
        
        # 机器学习工具
        ("scikit-learn", "机器学习算法"),
        
        # 图像处理和评估
        ("lpips", "感知损失计算"),
        ("pillow", "图像处理"),
        
        # 数据处理
        ("pyyaml", "YAML配置文件解析"),
        ("tqdm", "进度条"),
        
        # 可视化
        ("matplotlib", "绘图库"),
        ("seaborn", "统计绘图"),
        ("tensorboard", "TensorBoard日志"),
        
        # 其他工具
        ("einops", "张量操作"),
        ("safetensors", "安全张量保存"),
        ("fairscale", "分布式训练库"),
    ]
    
    # 安装依赖
    failed_packages = []
    for package, description in dependencies:
        package_name = package.split('>=')[0].split('==')[0]
        
        if check_package(package_name):
            print(f"✅ {package_name} 已安装")
        else:
            if not install_package(package, description):
                failed_packages.append(package)
    
    # 检查taming-transformers（VA-VAE需要）
    print("\n🔧 设置taming-transformers...")
    taming_path = Path("/kaggle/working/taming-transformers")
    
    if not taming_path.exists():
        print("📥 克隆taming-transformers...")
        try:
            subprocess.check_call([
                "git", "clone", 
                "https://github.com/CompVis/taming-transformers.git",
                str(taming_path)
            ])
            print("✅ taming-transformers 克隆成功")
        except subprocess.CalledProcessError:
            print("❌ taming-transformers 克隆失败")
            failed_packages.append("taming-transformers")
    else:
        print("✅ taming-transformers 已存在")
    
    # 创建路径文件
    path_file = Path("/kaggle/working/.taming_path")
    with open(path_file, 'w') as f:
        f.write(str(taming_path))
    print(f"📝 创建路径文件: {path_file}")
    
    # 总结
    print("\n" + "=" * 50)
    if failed_packages:
        print("❌ 以下包安装失败:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("请手动安装这些包或检查网络连接")
        return False
    else:
        print("🎉 所有依赖安装完成！")
        print("\n✅ 环境准备就绪，可以开始训练LightningDiT-S模型")
        
        # 显示下一步
        print("\n📝 Kaggle环境下一步:")
        print("1. 提取训练集特征: python extract_microdoppler_features.py --split train")
        print("2. 提取验证集特征: python extract_microdoppler_features.py --split val") 
        print("3. 开始DiT-S训练: python train_dit_s.py --config configs/dit_s_microdoppler.yaml")
        print("4. 注意：所有操作都在Kaggle Notebook中进行，无需本地测试")
        
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
