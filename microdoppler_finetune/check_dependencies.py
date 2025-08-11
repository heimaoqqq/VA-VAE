#!/usr/bin/env python3
"""
依赖检查脚本 - 确保所有必要的包都已安装
特别是 LPIPS，它对感知损失至关重要
"""

import sys
import subprocess
from pathlib import Path

def check_and_install_lpips():
    """检查并安装LPIPS包"""
    try:
        import lpips
        print("✅ LPIPS已安装")
        return True
    except ImportError:
        print("⚠️ LPIPS未安装，这对VA-VAE训练非常重要！")
        print("请在您的训练环境中运行：")
        print("pip install lpips")
        print("\n如果在Kaggle环境中，可以在notebook中运行：")
        print("!pip install lpips")
        return False

def check_taming():
    """检查taming-transformers"""
    taming_paths = [
        Path('/kaggle/working/taming-transformers'),
        Path.cwd().parent / 'taming-transformers',
    ]
    
    for path in taming_paths:
        if path.exists():
            print(f"✅ taming-transformers找到于: {path}")
            return True
    
    print("⚠️ taming-transformers未找到")
    print("虽然有回退实现，但建议安装以获得最佳性能")
    return False

def check_pytorch():
    """检查PyTorch"""
    try:
        import torch
        print(f"✅ PyTorch已安装: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️ CUDA不可用，训练会很慢")
        return True
    except ImportError:
        print("❌ PyTorch未安装！")
        return False

def main():
    print("=" * 60)
    print("VA-VAE 训练依赖检查")
    print("=" * 60)
    
    all_ok = True
    
    # 检查核心依赖
    if not check_pytorch():
        all_ok = False
    
    # 检查LPIPS（关键！）
    if not check_and_install_lpips():
        all_ok = False
        print("\n⚠️ 警告：没有LPIPS，感知损失将无法正常工作！")
        print("这会严重影响训练效果，因为perceptual_weight=1.0")
    
    # 检查taming（可选但推荐）
    check_taming()
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 所有关键依赖都已就绪！")
    else:
        print("⚠️ 缺少关键依赖，请在训练环境中安装")
        print("\n建议在Kaggle/Colab等环境中添加：")
        print("!pip install lpips")
        print("!git clone https://github.com/CompVis/taming-transformers.git")
    print("=" * 60)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
