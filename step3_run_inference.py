#!/usr/bin/env python3
"""
步骤3: 运行推理
严格按照LightningDiT README方法
"""

import os
import subprocess
import sys

def check_environment():
    """检查环境依赖"""
    
    print("🔍 检查环境依赖...")
    
    # 检查PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查Accelerate
    try:
        import accelerate
        print(f"✅ Accelerate: {accelerate.__version__}")
    except ImportError:
        print("❌ Accelerate未安装")
        print("💡 请安装: pip install accelerate")
        return False
    
    return True

def check_files():
    """检查必要文件"""
    
    print("\n📁 检查必要文件...")
    
    # 检查配置文件
    config_file = "inference_config.yaml"
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("💡 请先运行: python step2_setup_configs.py")
        return False
    
    print(f"✅ 推理配置: {config_file}")
    
    # 检查LightningDiT目录
    if not os.path.exists("LightningDiT"):
        print("❌ LightningDiT目录不存在")
        return False
    
    # 检查推理脚本
    inference_script = "LightningDiT/run_fast_inference.sh"
    if not os.path.exists(inference_script):
        print(f"❌ 推理脚本不存在: {inference_script}")
        return False
    
    print(f"✅ 推理脚本: {inference_script}")
    
    return True

def run_official_inference():
    """运行官方推理脚本"""
    
    print("\n🚀 运行官方推理脚本")
    print("-" * 30)
    
    # 切换到LightningDiT目录
    original_dir = os.getcwd()
    os.chdir("LightningDiT")
    
    try:
        # 构建官方命令
        config_path = "../inference_config.yaml"
        cmd = f"bash run_fast_inference.sh {config_path}"
        
        print(f"🎯 执行官方命令: {cmd}")
        print("📝 注意: 这是官方README中的标准命令")
        print("⏳ 推理中，请稍候...")
        
        # 运行命令
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=False, text=True)
        
        print("\n✅ 推理完成!")
        print("📁 生成的图像保存在: LightningDiT/demo_images/demo_samples.png")
        print("🎨 Demo模式参数:")
        print("   - cfg_scale: 9.0 (Demo模式自动设置)")
        print("   - 采样步数: 250")
        print("   - 采样方法: Euler")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 推理失败: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 检查是否安装了accelerate: pip install accelerate")
        print("2. 检查CUDA环境是否正常")
        print("3. 检查模型文件是否完整下载")
        print("4. 检查配置文件路径是否正确")
        return False
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断推理")
        return False
        
    finally:
        # 切换回原目录
        os.chdir(original_dir)

def main():
    """步骤3: 运行推理"""
    
    print("🚀 步骤3: 运行推理")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败")
        return
    
    # 检查文件
    if not check_files():
        print("\n❌ 文件检查失败")
        return
    
    # 运行推理
    if run_official_inference():
        print("\n🎉 步骤3完成！")
        print("📸 查看生成的图像: LightningDiT/demo_images/demo_samples.png")
        print("\n✅ 所有步骤完成！您已成功运行LightningDiT官方推理")
    else:
        print("\n❌ 推理失败")

if __name__ == "__main__":
    main()
