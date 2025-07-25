#!/usr/bin/env python3
"""
步骤1: 下载官方预训练模型
严格按照LightningDiT README步骤
"""

import os
import requests
from pathlib import Path

def download_file(url, local_path):
    """下载文件"""
    print(f"📥 下载: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r进度: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ 下载完成: {local_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def main():
    """步骤1: 下载官方预训练模型"""
    
    print("📥 步骤1: 下载官方预训练模型")
    print("=" * 50)
    
    # 创建模型目录
    models_dir = Path("./official_models")
    models_dir.mkdir(exist_ok=True)
    
    # 官方README中的下载链接
    models = [
        {
            "name": "VA-VAE Tokenizer",
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt",
            "filename": "vavae-imagenet256-f16d32-dinov2.pt"
        },
        {
            "name": "LightningDiT-XL-800ep",
            "url": "https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/resolve/main/lightningdit-xl-imagenet256-800ep.pt",
            "filename": "lightningdit-xl-imagenet256-800ep.pt"
        },
        {
            "name": "Latent Statistics",
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt",
            "filename": "latents_stats.pt"
        }
    ]
    
    success_count = 0
    
    for model in models:
        filepath = models_dir / model["filename"]
        
        if filepath.exists():
            print(f"✅ {model['name']}: 已存在 ({filepath.stat().st_size / (1024*1024):.1f} MB)")
            success_count += 1
        else:
            print(f"\n📥 下载 {model['name']}...")
            if download_file(model["url"], str(filepath)):
                success_count += 1
    
    print(f"\n📊 下载结果: {success_count}/{len(models)} 个文件成功")
    
    if success_count == len(models):
        print("✅ 步骤1完成！所有模型文件已下载")
        print(f"📁 模型位置: {models_dir.absolute()}")
        print("\n🎯 下一步: 运行 python step2_setup_configs.py")
    else:
        print("❌ 部分文件下载失败，请检查网络连接")

if __name__ == "__main__":
    main()
