#!/usr/bin/env python3
"""
修复DiT训练依赖问题
安装缺失的包并验证导入
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ 成功: {description}")
            return True, result.stdout
        else:
            print(f"❌ 失败: {description}")
            print(f"错误信息: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return False, str(e)

def install_missing_packages():
    """安装缺失的依赖包"""
    print("🔧 安装DiT训练所需的依赖包...")
    
    packages = [
        "fairscale>=0.4.13",
        "einops>=0.6.0", 
        "timm>=0.9.0",
        "torchdiffeq>=0.2.3",
        "omegaconf>=2.3.0",
        "diffusers>=0.20.0",
        "pytorch-fid>=0.3.0",
        "scipy>=1.9.0",
        "tensorboard>=2.16.0"
    ]
    
    success_count = 0
    for package in packages:
        print(f"\n📦 安装: {package}")
        success, _ = run_command(f"pip install {package}", f"安装 {package}")
        if success:
            success_count += 1
    
    print(f"\n✅ 成功安装 {success_count}/{len(packages)} 个包")
    return success_count == len(packages)

def test_imports():
    """测试关键导入"""
    print("\n🧪 测试关键导入...")
    
    test_cases = [
        ("torch", "PyTorch"),
        ("fairscale", "FairScale"),
        ("einops", "Einops"),
        ("timm", "TIMM"),
        ("torchdiffeq", "TorchDiffEq"),
        ("omegaconf", "OmegaConf"),
        ("diffusers", "Diffusers"),
        ("pytorch_fid", "PyTorch FID"),
        ("scipy", "SciPy"),
        ("tensorboard", "TensorBoard")
    ]
    
    success_count = 0
    for module, name in test_cases:
        try:
            __import__(module)
            print(f"✅ {name}: 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: 导入失败 - {e}")
    
    print(f"\n✅ 成功导入 {success_count}/{len(test_cases)} 个模块")
    return success_count == len(test_cases)

def test_rmsnorm():
    """测试RMSNorm导入"""
    print("\n🧪 测试RMSNorm导入...")
    
    try:
        # 测试fairscale版本
        from fairscale.nn.model_parallel.initialize import initialize_model_parallel
        print("✅ FairScale RMSNorm可用")
        return True
    except ImportError:
        print("⚠️  FairScale RMSNorm不可用，使用简化版本")
        
        try:
            # 测试简化版本
            from simple_rmsnorm import RMSNorm
            print("✅ 简化版RMSNorm可用")
            return True
        except ImportError:
            print("❌ 简化版RMSNorm也不可用")
            return False

def test_lightningdit_import():
    """测试LightningDiT导入"""
    print("\n🧪 测试LightningDiT导入...")
    
    try:
        import sys
        sys.path.append('LightningDiT')
        
        from models.lightningdit import LightningDiT_models
        print("✅ LightningDiT模型导入成功")
        
        from transport import create_transport
        print("✅ Transport模块导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ LightningDiT导入失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 修复DiT训练依赖问题")
    print("=" * 50)
    
    # 安装依赖包
    if not install_missing_packages():
        print("❌ 依赖包安装失败")
        return False
    
    # 测试导入
    if not test_imports():
        print("❌ 基础导入测试失败")
        return False
    
    # 测试RMSNorm
    if not test_rmsnorm():
        print("❌ RMSNorm测试失败")
        return False
    
    # 测试LightningDiT
    if not test_lightningdit_import():
        print("❌ LightningDiT导入测试失败")
        return False
    
    print("\n🎉 所有依赖问题已修复!")
    print("现在可以开始DiT训练了:")
    print("  python kaggle_training_wrapper.py stage2")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 修复失败，请检查错误信息")
        sys.exit(1)
