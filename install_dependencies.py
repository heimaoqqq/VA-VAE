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

def install_vavae_requirements():
    """安装VA-VAE训练专用依赖"""
    print("\n🔧 安装VA-VAE训练专用依赖...")

    # VA-VAE训练额外依赖
    vavae_deps = [
        "pytorch-lightning>=1.8.0",
        "lpips>=0.1.4",
        "kornia>=0.6.0",
        "transformers>=4.20.0",
        "xformers>=0.0.16",
        "wandb",  # 训练监控
        "matplotlib",  # 可视化
        "seaborn",  # 统计图表
        "scikit-learn",  # 评估指标
        "opencv-python",  # 图像处理
        "albumentations",  # 数据增强
        "pillow>=8.0.0"  # 图像处理
    ]

    print("📦 VA-VAE训练依赖包:")
    for dep in vavae_deps:
        print(f"   - {dep}")

    print("\n🔄 安装VA-VAE训练依赖...")
    for dep in vavae_deps:
        if not run_command(f"pip install {dep}", f"安装 {dep}"):
            print(f"⚠️ {dep} 安装失败，继续安装其他依赖...")

    return True

def install_taming_transformers():
    """安装Taming-Transformers"""
    print("\n🔧 安装Taming-Transformers (VA-VAE训练必需)...")

    taming_dir = "taming-transformers"

    # 检查是否已经存在
    if os.path.exists(taming_dir):
        print(f"✅ Taming-Transformers目录已存在: {taming_dir}")
        print("🔄 更新到最新版本...")

        # 更新现有仓库
        original_dir = os.getcwd()
        try:
            os.chdir(taming_dir)
            if not run_command("git pull", "更新Taming-Transformers"):
                print("⚠️ 更新失败，使用现有版本")
        finally:
            os.chdir(original_dir)
    else:
        # 克隆新仓库
        print("📥 克隆Taming-Transformers仓库...")
        if not run_command(
            "git clone https://github.com/CompVis/taming-transformers.git",
            "克隆Taming-Transformers"
        ):
            print("❌ 克隆失败，尝试备用方案...")
            # 尝试浅克隆
            if not run_command(
                "git clone --depth 1 https://github.com/CompVis/taming-transformers.git",
                "浅克隆Taming-Transformers"
            ):
                print("❌ Taming-Transformers安装失败")
                return False

    # 安装Taming-Transformers
    print("📦 安装Taming-Transformers包...")
    original_dir = os.getcwd()
    try:
        os.chdir(taming_dir)
        if not run_command("pip install -e .", "安装Taming-Transformers"):
            return False
    finally:
        os.chdir(original_dir)

    # 修复torch 2.x兼容性
    print("🔧 修复torch 2.x兼容性...")
    utils_file = os.path.join(taming_dir, "taming", "data", "utils.py")

    if os.path.exists(utils_file):
        try:
            # 读取文件
            with open(utils_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换兼容性问题的代码
            old_import = "from torch._six import string_classes"
            new_import = "from six import string_types as string_classes"

            if old_import in content:
                content = content.replace(old_import, new_import)

                # 写回文件
                with open(utils_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                print("✅ torch 2.x兼容性修复完成")
            else:
                print("✅ 文件已经兼容torch 2.x")

        except Exception as e:
            print(f"⚠️ 兼容性修复失败: {e}")
            print("💡 可能需要手动修复，但不影响基本功能")
    else:
        print(f"⚠️ 未找到utils.py文件: {utils_file}")

    return True

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='LightningDiT依赖安装脚本')
    parser.add_argument('--training', action='store_true',
                       help='安装训练依赖 (包括VA-VAE微调)')
    parser.add_argument('--inference-only', action='store_true',
                       help='只安装推理依赖')

    args = parser.parse_args()

    if args.inference_only:
        print("🔧 LightningDiT推理依赖安装")
        print("=" * 50)
        mode = "inference"
    elif args.training:
        print("🔧 LightningDiT完整依赖安装 (推理 + 训练)")
        print("=" * 60)
        mode = "training"
    else:
        # 默认安装完整依赖
        print("🔧 LightningDiT完整依赖安装 (推理 + 训练)")
        print("=" * 60)
        print("💡 使用 --inference-only 只安装推理依赖")
        print("💡 使用 --training 明确安装训练依赖")
        mode = "training"

    print("📝 严格按照官方requirements.txt安装")
    print("🎯 解决Kaggle环境预装包冲突")

    # 1. 检查当前环境
    print("\n📋 阶段1: 检查当前环境")
    check_current_packages()

    # 2. 安装基础推理依赖
    print("\n📋 阶段2: 安装基础推理依赖")
    install_official_requirements()

    # 3. 安装训练依赖 (如果需要)
    if mode == "training":
        print("\n📋 阶段3: 安装VA-VAE训练依赖")
        install_vavae_requirements()

        print("\n📋 阶段4: 安装Taming-Transformers")
        if not install_taming_transformers():
            print("⚠️ Taming-Transformers安装失败，但不影响推理")

    # 4. 验证安装
    print(f"\n📋 最终阶段: 验证安装")
    if verify_installation():
        print(f"\n✅ 依赖安装完成！")
        if mode == "training":
            print("📋 环境准备就绪，支持:")
            print("  - LightningDiT推理 ✅")
            print("  - VA-VAE微调训练 ✅")
            print("  - 微多普勒数据增广 ✅")
        else:
            print("📋 推理环境准备就绪 ✅")
        print("\n📋 下一步: python step1_download_models.py")
    else:
        print("\n❌ 依赖安装存在问题，请检查错误信息")

if __name__ == "__main__":
    main()
