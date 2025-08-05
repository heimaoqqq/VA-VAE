#!/usr/bin/env python3
"""
快速环境检查脚本
用于验证LightningDiT复现环境是否正确配置
"""

import sys
from pathlib import Path

def check_basic_environment():
    """检查基础环境"""
    print("🔍 快速环境检查")
    print("="*40)
    
    # 检查Python版本
    python_version = sys.version
    print(f"🐍 Python版本: {python_version}")
    
    # 检查关键模块
    modules_status = {}
    critical_modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('accelerate', 'Accelerate'),
        ('torchdiffeq', 'TorchDiffEq')
    ]
    
    for module, name in critical_modules:
        try:
            __import__(module)
            modules_status[name] = "✅"
        except ImportError:
            modules_status[name] = "❌"
        except Exception:
            modules_status[name] = "⚠️"  # 警告但可用
    
    # 显示模块状态
    print("\n📦 关键模块状态:")
    for name, status in modules_status.items():
        print(f"   {status} {name}")
    
    # 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\n🔥 CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"🔥 GPU数量: {torch.cuda.device_count()}")
    except:
        print("\n❌ 无法检查CUDA状态")
    
    # 检查项目文件
    print("\n📁 项目文件检查:")
    required_files = [
        "step1_install_environment.py",
        "step2_download_models.py", 
        "step3_setup_configs.py",
        "step4_inference.py",
        "LightningDiT/inference.py"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            all_files_exist = False
    
    # 总结
    print("\n" + "="*40)
    critical_modules_ok = all(status != "❌" for status in modules_status.values())
    
    if critical_modules_ok and all_files_exist:
        print("🎉 环境检查通过！可以开始复现")
        print("📋 执行顺序:")
        print("   1. !python step1_install_environment.py")
        print("   2. !python step2_download_models.py")
        print("   3. !python step3_setup_configs.py")
        print("   4. !python step4_inference.py")
        return True
    else:
        print("⚠️ 环境存在问题，请检查:")
        if not critical_modules_ok:
            print("   - 关键模块缺失或损坏")
        if not all_files_exist:
            print("   - 项目文件不完整")
        return False

if __name__ == "__main__":
    success = check_basic_environment()
    sys.exit(0 if success else 1)
