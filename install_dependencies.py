#!/usr/bin/env python3
"""
VA-VAE依赖安装脚本 - 专为Kaggle环境优化
按照官方LightningDiT项目的要求安装所有必要依赖
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"🔧 {description}")
    print(f"   执行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败")
        print(f"   错误: {e.stderr}")
        return False

def install_taming_transformers():
    """安装taming-transformers - 官方方式"""
    print("\n🚀 安装 taming-transformers...")
    
    # 检查是否已经存在
    if Path("taming-transformers").exists():
        print("📁 taming-transformers 目录已存在，跳过克隆")
    else:
        # 克隆仓库
        if not run_command(
            "git clone https://github.com/CompVis/taming-transformers.git",
            "克隆 taming-transformers 仓库"
        ):
            return False
    
    # 进入目录并安装
    original_dir = os.getcwd()
    try:
        os.chdir("taming-transformers")
        
        # 修复torch 2.x兼容性问题
        utils_file = Path("taming/data/utils.py")
        if utils_file.exists():
            print("🔧 修复torch 2.x兼容性...")
            with open(utils_file, 'r') as f:
                content = f.read()
            
            # 替换过时的导入
            if "from torch._six import string_classes" in content:
                content = content.replace(
                    "from torch._six import string_classes",
                    "from six import string_types as string_classes"
                )
                with open(utils_file, 'w') as f:
                    f.write(content)
                print("✅ torch 2.x兼容性修复完成")
            else:
                print("ℹ️ 兼容性已修复或不需要修复")
        
        # 安装包
        if not run_command(
            f"{sys.executable} -m pip install -e .",
            "安装 taming-transformers"
        ):
            return False
            
    finally:
        os.chdir(original_dir)
    
    return True

def install_other_dependencies():
    """安装其他必要依赖"""
    print("\n🚀 安装其他依赖...")
    
    dependencies = [
        "pytorch-lightning",
        "omegaconf", 
        "einops",
        "transformers",
        "accelerate"
    ]
    
    for dep in dependencies:
        if not run_command(
            f"{sys.executable} -m pip install {dep}",
            f"安装 {dep}"
        ):
            print(f"⚠️ {dep} 安装失败，但继续安装其他依赖")
    
    return True

def verify_installation():
    """验证安装结果"""
    print("\n🔍 验证安装...")
    
    # 测试taming-transformers
    try:
        import taming.data.utils as tdu
        import taming.modules.losses.vqperceptual
        from taming.modules.vqvae.quantize import VectorQuantizer2
        print("✅ taming-transformers 验证成功")
    except ImportError as e:
        print(f"❌ taming-transformers 验证失败: {e}")
        return False
    
    # 测试pytorch-lightning
    try:
        import pytorch_lightning as pl
        print(f"✅ pytorch-lightning {pl.__version__} 验证成功")
    except ImportError as e:
        print(f"❌ pytorch-lightning 验证失败: {e}")
        return False
    
    # 测试其他依赖
    try:
        import omegaconf
        import einops
        print("✅ 其他依赖验证成功")
    except ImportError as e:
        print(f"❌ 其他依赖验证失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 VA-VAE 依赖安装脚本")
    print("   专为Kaggle环境优化")
    print("=" * 60)
    
    # 安装taming-transformers
    if not install_taming_transformers():
        print("❌ taming-transformers 安装失败")
        return False
    
    # 安装其他依赖
    install_other_dependencies()
    
    # 验证安装
    if verify_installation():
        print("\n🎉 所有依赖安装成功！")
        print("💡 现在可以运行: python finetune_vavae.py")
        return True
    else:
        print("\n❌ 依赖验证失败")
        print("💡 请检查错误信息并手动解决")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
