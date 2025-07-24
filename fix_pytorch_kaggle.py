#!/usr/bin/env python3
"""
Kaggle PyTorch修复脚本
专门解决torch._C模块冲突问题
"""

import subprocess
import sys
import os

def print_step(step, text):
    print(f"\n🔧 Step {step}: {text}")
    print("-" * 50)

def run_command(cmd, description=""):
    """运行命令"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ 成功: {description}")
            return True, result.stdout
        else:
            print(f"❌ 失败: {description}")
            print(f"错误: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return False, str(e)

def main():
    print("🎯 Kaggle PyTorch修复脚本")
    print("=" * 60)
    
    print_step(1, "检测PyTorch状态")
    try:
        import torch
        print(f"当前PyTorch版本: {torch.__version__}")
        try:
            import torch._C
            print("✅ torch._C模块正常")
            print("🎉 PyTorch工作正常，无需修复")
            return
        except Exception as e:
            print(f"❌ torch._C模块错误: {e}")
    except Exception as e:
        print(f"❌ PyTorch导入错误: {e}")
    
    print_step(2, "完全清理PyTorch")
    cleanup_commands = [
        "pip uninstall torch torchvision torchaudio -y",
        "pip uninstall torch-audio torch-vision -y",
        "pip cache purge"
    ]
    
    for cmd in cleanup_commands:
        run_command(cmd, f"执行清理: {cmd}")
    
    print_step(3, "安装稳定版PyTorch")
    # 使用Kaggle兼容的稳定版本
    install_cmd = "pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118"
    success, _ = run_command(install_cmd, "安装PyTorch")
    
    if success:
        print_step(4, "验证安装")
        print("🔄 请重启Kaggle内核，然后运行以下代码验证:")
        print("""
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

# 测试基本功能
x = torch.randn(2, 3)
print(f"张量创建成功: {x.shape}")

# 测试C++扩展
import torch._C
print("✅ torch._C模块正常")
        """)
        
        print("\n🎉 修复完成！请重启内核验证")
    else:
        print("❌ 修复失败，请手动重启内核后重试")

if __name__ == "__main__":
    main()
