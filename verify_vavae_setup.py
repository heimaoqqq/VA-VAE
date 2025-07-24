#!/usr/bin/env python3
"""
验证VA-VAE设置脚本
检查模型文件、配置文件和加载功能
"""

import os
from pathlib import Path
import torch

def check_model_files():
    """检查模型文件是否存在"""
    print("🔍 检查模型文件...")
    
    model_path = Path("/kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt")
    stats_path = Path("/kaggle/working/pretrained/latents_stats.pt")
    
    # 检查VA-VAE模型
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ VA-VAE模型存在: {model_path}")
        print(f"   文件大小: {file_size:.1f}MB")
        
        if file_size < 100:
            print("⚠️  文件大小异常，可能下载不完整")
            return False
    else:
        print(f"❌ VA-VAE模型不存在: {model_path}")
        return False
    
    # 检查统计信息文件
    if stats_path.exists():
        file_size = stats_path.stat().st_size / 1024  # KB
        print(f"✅ 统计信息文件存在: {stats_path}")
        print(f"   文件大小: {file_size:.1f}KB")
    else:
        print(f"⚠️  统计信息文件不存在: {stats_path}")
        print("   这不会影响训练，但可能影响性能")
    
    return True

def check_config_file():
    """检查配置文件"""
    print("\n🔍 检查配置文件...")
    
    config_path = Path("vavae_config.yaml")
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    # 读取配置内容
    with open(config_path, 'r') as f:
        content = f.read()
    
    print(f"✅ 配置文件存在: {config_path}")
    
    # 检查关键配置
    if 'ckpt_path:' in content:
        print("✅ 找到 ckpt_path 配置")
        
        # 提取路径
        for line in content.split('\n'):
            if 'ckpt_path:' in line:
                path = line.split(':', 1)[1].strip().strip('"')
                print(f"   配置路径: {path}")
                
                # 检查路径是否存在
                if Path(path).exists():
                    print("✅ 配置路径指向的文件存在")
                else:
                    print("❌ 配置路径指向的文件不存在")
                    return False
                break
    else:
        print("❌ 缺少 ckpt_path 配置")
        return False
    
    return True

def test_vavae_loading():
    """测试VA-VAE加载"""
    print("\n🧪 测试VA-VAE加载...")
    
    try:
        # 添加LightningDiT路径
        import sys
        sys.path.append('LightningDiT')
        
        # 导入VA-VAE
        from tokenizer.vavae import VA_VAE
        
        print("✅ VA-VAE类导入成功")
        
        # 尝试初始化
        print("🔄 初始化VA-VAE...")
        vavae = VA_VAE('vavae_config.yaml')
        print("✅ VA-VAE初始化成功!")
        
        # 测试编码功能
        print("🔄 测试编码功能...")
        test_image = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            encoded = vavae.encode(test_image)
            latent = encoded.sample()
            print(f"✅ 编码测试成功!")
            print(f"   输入形状: {test_image.shape}")
            print(f"   输出形状: {latent.shape}")
            print(f"   预期形状: torch.Size([1, 32, 16, 16])")
            
            if latent.shape == torch.Size([1, 32, 16, 16]):
                print("✅ 输出形状正确")
            else:
                print("⚠️  输出形状异常")
        
        return True
        
    except Exception as e:
        print(f"❌ VA-VAE测试失败: {e}")
        print("\n详细错误信息:")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n🔍 检查依赖包...")
    
    required_packages = [
        'torch',
        'omegaconf',
        'accelerate',
        'safetensors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少依赖包: {missing_packages}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """主函数"""
    print("🎯 VA-VAE设置验证")
    print("=" * 50)
    
    all_checks_passed = True
    
    # 检查依赖
    if not check_dependencies():
        all_checks_passed = False
    
    # 检查模型文件
    if not check_model_files():
        all_checks_passed = False
    
    # 检查配置文件
    if not check_config_file():
        all_checks_passed = False
    
    # 测试加载
    if not test_vavae_loading():
        all_checks_passed = False
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 所有检查通过! VA-VAE设置正确")
        print("现在可以开始双GPU特征提取了:")
        print("  python kaggle_training_wrapper.py stage1")
    else:
        print("❌ 部分检查失败，请修复上述问题")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
