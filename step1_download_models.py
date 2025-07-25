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
    """步骤1: 下载预训练模型"""
    import argparse

    parser = argparse.ArgumentParser(description='下载LightningDiT预训练模型')
    parser.add_argument('--inference-only', action='store_true',
                       help='只下载推理所需模型 (全部3个)')
    parser.add_argument('--training-only', action='store_true',
                       help='下载微多普勒训练所需模型 (仅VA-VAE)')
    parser.add_argument('--minimal', action='store_true',
                       help='最小下载 (只下载VA-VAE，约800MB)')
    parser.add_argument('--vae-only', action='store_true',
                       help='仅下载VA-VAE模型 (微多普勒训练推荐)')

    args = parser.parse_args()

    if args.training_only or args.vae_only:
        print("📥 步骤1: 下载微多普勒训练模型 (仅VA-VAE)")
        print("=" * 60)
        print("🎯 专为微多普勒训练优化：只下载必需的VA-VAE模型")
        mode = "vae_only"
    elif args.minimal:
        print("📥 步骤1: 最小下载 (仅VA-VAE)")
        print("=" * 50)
        mode = "vae_only"
    else:
        print("📥 步骤1: 下载完整预训练模型")
        print("=" * 60)
        print("💡 使用 --vae-only 只下载VA-VAE (微多普勒训练推荐)")
        print("💡 使用 --minimal 最小下载 (约800MB)")
        mode = "full"

    # 创建模型目录
    models_dir = Path("./official_models")
    models_dir.mkdir(exist_ok=True)

    # 定义所有可用模型
    all_models = [
        {
            "name": "VA-VAE Tokenizer",
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt",
            "filename": "vavae-imagenet256-f16d32-dinov2.pt",
            "size": "~800MB",
            "required_for": ["inference", "training", "minimal", "vae_only"],
            "description": "VA-VAE编码器/解码器，微多普勒训练的基础模型"
        },
        {
            "name": "Latent Statistics (ImageNet)",
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt",
            "filename": "latents_stats.pt",
            "size": "~1KB",
            "required_for": ["inference"],
            "description": "ImageNet潜在特征统计，微多普勒训练时会重新计算"
        },
        {
            "name": "LightningDiT-XL-800ep",
            "url": "https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/resolve/main/lightningdit-xl-imagenet256-800ep.pt",
            "filename": "lightningdit-xl-imagenet256-800ep.pt",
            "size": "~6GB",
            "required_for": ["inference"],
            "description": "预训练扩散模型，用于推理演示"
        }
    ]

    # 根据模式选择要下载的模型
    if mode == "vae_only":
        models = [m for m in all_models if "vae_only" in m["required_for"]]
    elif mode == "training":
        models = [m for m in all_models if "training" in m["required_for"]]
    else:  # full
        models = all_models

    # 显示下载计划
    print(f"\n📋 下载计划 ({mode} 模式):")
    total_size_info = []
    for model in models:
        print(f"  ✅ {model['name']} ({model['size']})")
        print(f"     {model['description']}")
        if model['size'].replace('~', '').replace('MB', '').replace('GB', '').replace('KB', '').isdigit():
            if 'GB' in model['size']:
                total_size_info.append(float(model['size'].replace('~', '').replace('GB', '')) * 1000)
            elif 'MB' in model['size']:
                total_size_info.append(float(model['size'].replace('~', '').replace('MB', '')))

    if total_size_info:
        total_mb = sum(total_size_info)
        if total_mb > 1000:
            print(f"\n📊 预计总大小: ~{total_mb/1000:.1f}GB")
        else:
            print(f"\n📊 预计总大小: ~{total_mb:.0f}MB")
    
    success_count = 0

    for model in models:
        filepath = models_dir / model["filename"]

        if filepath.exists():
            print(f"✅ {model['name']}: 已存在 ({filepath.stat().st_size / (1024*1024):.1f} MB)")
            success_count += 1
        else:
            print(f"\n📥 下载 {model['name']} ({model['size']})...")
            if download_file(model["url"], str(filepath)):
                success_count += 1

    print(f"\n📊 下载结果: {success_count}/{len(models)} 个文件成功")

    if success_count == len(models):
        print("✅ 步骤1完成！模型文件已下载")
        print(f"📁 模型位置: {models_dir.absolute()}")

        # 根据模式给出不同的下一步建议
        if mode == "vae_only":
            print("\n🎯 微多普勒训练模型已下载！")
            print("✅ VA-VAE模型 - 微调训练的基础")
            print("💡 说明：")
            print("   - latents_stats.pt (ImageNet统计) 已跳过")
            print("   - 训练时会基于您的数据重新计算统计信息")
            print("   - LightningDiT扩散模型将重头训练")
            print("📋 下一步: python step3_prepare_micro_doppler_dataset.py")
        elif mode == "training":
            print("\n🎯 微多普勒训练准备就绪！")
            print("📋 下一步: python step3_prepare_micro_doppler_dataset.py")
        else:
            print("\n🎯 完整模型已下载！")
            print("📋 推理测试: python step2_setup_configs.py")
            print("📋 微多普勒训练: python step3_prepare_micro_doppler_dataset.py")
    else:
        print("❌ 部分文件下载失败，请检查网络连接")
        print("💡 可以尝试:")
        print("   - 使用 --minimal 只下载VA-VAE (约800MB)")
        print("   - 使用 --training-only 只下载训练所需模型")

if __name__ == "__main__":
    main()
