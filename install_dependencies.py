#!/usr/bin/env python3
"""
严格按照LightningDiT官方requirements.txt安装依赖
处理Kaggle环境的预装包冲突问题
"""

import subprocess
import sys
import pkg_resources

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    print(f"🔧 {description}")
    print(f"💻 执行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ 成功")
        if result.stdout:
            print(f"输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        if e.stderr:
            print(f"错误: {e.stderr.strip()}")
        return False

def check_current_packages():
    """检查当前已安装的包"""
    print("📋 检查当前已安装的关键包...")
    
    key_packages = [
        'torch', 'torchvision', 'accelerate', 'torchdiffeq', 
        'timm', 'diffusers', 'pytorch_fid', 'tensorboard',
        'omegaconf', 'einops', 'fairscale', 'safetensors'
    ]
    
    installed = {}
    for package in key_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            installed[package] = version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            installed[package] = None
            print(f"❌ {package}: 未安装")
    
    return installed

def install_official_requirements():
    """按照官方requirements.txt安装依赖"""
    
    print("\n🎯 按照LightningDiT官方requirements.txt安装依赖")
    print("=" * 60)
    
    # 官方requirements.txt的内容
    official_requirements = [
        # PyTorch相关 - 根据CUDA版本调整
        "torch==2.2.0",  # 移除+cu121，让pip自动选择合适版本
        "torchvision==0.17.0",
        
        # 核心依赖
        "timm==1.0.12",
        "diffusers==0.32.1", 
        "accelerate",
        "torchdiffeq",  # 这是缺失的关键依赖
        "pytorch_fid",
        "tensorboard==2.16.2",
        "omegaconf==2.3.0",
        "einops",
        "fairscale",
        "safetensors"
    ]
    
    print("📦 官方要求的依赖包:")
    for req in official_requirements:
        print(f"   - {req}")
    
    # 在Kaggle环境中，先卸载可能冲突的包
    print("\n🔄 处理Kaggle预装包冲突...")
    conflicting_packages = ['torch', 'torchvision', 'accelerate']
    
    for package in conflicting_packages:
        print(f"\n📤 卸载现有的 {package}...")
        run_command(f"pip uninstall {package} -y", f"卸载 {package}")
    
    # 安装官方指定版本
    print("\n📥 安装官方指定版本...")
    
    # 分组安装，避免依赖冲突
    install_groups = [
        # 第一组：PyTorch核心
        ["torch==2.2.0", "torchvision==0.17.0"],
        
        # 第二组：关键缺失依赖
        ["torchdiffeq", "accelerate"],
        
        # 第三组：其他依赖
        ["timm==1.0.12", "diffusers==0.32.1", "pytorch_fid"],
        
        # 第四组：工具包
        ["tensorboard==2.16.2", "omegaconf==2.3.0", "einops", "fairscale", "safetensors"]
    ]
    
    for i, group in enumerate(install_groups, 1):
        print(f"\n📦 安装第{i}组依赖...")
        for package in group:
            if not run_command(f"pip install {package}", f"安装 {package}"):
                print(f"⚠️ {package} 安装失败，继续安装其他包...")
    
    print("\n✅ 依赖安装完成")

def verify_installation():
    """验证安装结果"""
    print("\n🔍 验证安装结果...")
    print("=" * 40)
    
    # 检查关键模块是否可以导入
    test_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"), 
        ("accelerate", "Accelerate"),
        ("torchdiffeq", "TorchDiffEq - 关键缺失模块"),
        ("timm", "TIMM"),
        ("diffusers", "Diffusers"),
        ("pytorch_fid", "PyTorch FID"),
        ("tensorboard", "TensorBoard"),
        ("omegaconf", "OmegaConf"),
        ("einops", "Einops"),
        ("fairscale", "FairScale"),
        ("safetensors", "SafeTensors")
    ]
    
    success_count = 0
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}: 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: 导入失败 - {e}")
    
    print(f"\n📊 验证结果: {success_count}/{len(test_imports)} 个模块成功")
    
    if success_count == len(test_imports):
        print("🎉 所有依赖安装成功！")
        print("🚀 现在可以运行: python step3_run_inference.py")
        return True
    else:
        print("⚠️ 部分依赖安装失败，可能影响推理")
        return False

def main():
    """主函数"""
    print("🔧 LightningDiT官方依赖安装脚本")
    print("=" * 50)
    print("📝 严格按照官方requirements.txt安装")
    print("🎯 解决Kaggle环境预装包冲突")
    
    # 检查当前环境
    check_current_packages()
    
    # 安装官方依赖
    install_official_requirements()
    
    # 验证安装
    if verify_installation():
        print("\n✅ 依赖安装完成！可以继续推理了")
    else:
        print("\n❌ 依赖安装存在问题，请检查错误信息")

if __name__ == "__main__":
    main()
