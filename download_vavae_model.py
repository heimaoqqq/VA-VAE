#!/usr/bin/env python3
"""
下载VA-VAE预训练模型脚本
确保模型文件存在且完整
"""

import os
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm

def download_file_with_progress(url, filepath):
    """带进度条的文件下载"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

def download_vavae_model():
    """下载VA-VAE预训练模型"""
    print("🔄 下载VA-VAE预训练模型...")
    
    # 创建目录
    pretrained_dir = Path("/kaggle/working/pretrained")
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = pretrained_dir / "vavae-imagenet256-f16d32-dinov2.pt"
    
    # 检查文件是否已存在
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 模型文件已存在: {model_path}")
        print(f"   文件大小: {file_size:.1f}MB")
        
        # 简单验证文件完整性
        if file_size > 100:  # 预期模型应该大于100MB
            print("✅ 文件大小正常，跳过下载")
            return True
        else:
            print("⚠️  文件大小异常，重新下载")
            model_path.unlink()
    
    # 下载模型
    model_url = "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt"
    
    print(f"📥 从 HuggingFace 下载模型...")
    print(f"   URL: {model_url}")
    print(f"   保存到: {model_path}")
    
    try:
        # 方法1: 使用requests下载
        download_file_with_progress(model_url, model_path)
        
        # 验证下载
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ 下载完成! 文件大小: {file_size:.1f}MB")
            return True
        else:
            print("❌ 下载失败，文件不存在")
            return False
            
    except Exception as e:
        print(f"❌ requests下载失败: {e}")
        
        # 方法2: 使用wget下载
        try:
            print("🔄 尝试使用wget下载...")
            result = subprocess.run([
                'wget', '-O', str(model_path), model_url
            ], capture_output=True, text=True, check=True)
            
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)
                print(f"✅ wget下载完成! 文件大小: {file_size:.1f}MB")
                return True
            else:
                print("❌ wget下载失败")
                return False
                
        except Exception as e2:
            print(f"❌ wget下载也失败: {e2}")
            
            # 方法3: 使用curl下载
            try:
                print("🔄 尝试使用curl下载...")
                result = subprocess.run([
                    'curl', '-L', '-o', str(model_path), model_url
                ], capture_output=True, text=True, check=True)
                
                if model_path.exists():
                    file_size = model_path.stat().st_size / (1024 * 1024)
                    print(f"✅ curl下载完成! 文件大小: {file_size:.1f}MB")
                    return True
                else:
                    print("❌ curl下载失败")
                    return False
                    
            except Exception as e3:
                print(f"❌ 所有下载方法都失败了")
                print(f"   requests错误: {e}")
                print(f"   wget错误: {e2}")
                print(f"   curl错误: {e3}")
                return False

def verify_vavae_config():
    """验证VA-VAE配置文件"""
    print("\n🔍 验证VA-VAE配置...")
    
    config_file = Path("vavae_config.yaml")
    if not config_file.exists():
        print("❌ vavae_config.yaml 不存在")
        return False
    
    # 读取配置文件
    with open(config_file, 'r') as f:
        content = f.read()
    
    # 检查关键配置
    if 'ckpt_path:' in content:
        print("✅ 找到 ckpt_path 配置")
    else:
        print("❌ 缺少 ckpt_path 配置")
        return False
    
    if '/kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt' in content:
        print("✅ 模型路径配置正确")
    else:
        print("❌ 模型路径配置错误")
        return False
    
    print("✅ VA-VAE配置验证通过")
    return True

def test_vavae_loading():
    """测试VA-VAE模型加载"""
    print("\n🧪 测试VA-VAE模型加载...")
    
    try:
        import sys
        sys.path.append('LightningDiT')
        
        from tokenizer.vavae import VA_VAE
        
        # 尝试加载模型
        vavae = VA_VAE('vavae_config.yaml')
        print("✅ VA-VAE模型加载成功!")
        
        # 测试编码功能
        import torch
        test_image = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            encoded = vavae.encode(test_image)
            print(f"✅ 编码测试成功! 输出形状: {encoded.sample().shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VA-VAE模型加载失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 VA-VAE模型下载和验证")
    print("=" * 50)
    
    # 下载模型
    if not download_vavae_model():
        print("❌ 模型下载失败，无法继续")
        return False
    
    # 验证配置
    if not verify_vavae_config():
        print("❌ 配置验证失败，无法继续")
        return False
    
    # 测试加载
    if not test_vavae_loading():
        print("❌ 模型加载测试失败")
        return False
    
    print("\n🎉 VA-VAE模型准备完成!")
    print("现在可以开始特征提取了")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 准备过程失败，请检查错误信息")
        exit(1)
