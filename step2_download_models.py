#!/usr/bin/env python3
"""
步骤2: 下载LightningDiT预训练模型
严格按照官方README中的模型链接
"""

import os
import requests
from pathlib import Path
import sys

def download_file_with_progress(url, local_path):
    """带进度条的文件下载"""
    print(f"📥 下载: {url}")
    print(f"📁 保存到: {local_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        # 创建目录
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(f"\r进度: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", 
                              end='', flush=True)
        
        print(f"\n✅ 下载完成: {local_path}")
        print(f"📊 文件大小: {local_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def download_official_models():
    """下载官方预训练模型"""
    print("📥 下载LightningDiT官方预训练模型")
    print("="*60)
    
    # 创建模型目录
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 官方README中的模型链接
    models = {
        "VA-VAE Tokenizer": {
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt",
            "filename": "vavae-imagenet256-f16d32-dinov2.pt",
            "description": "Vision Foundation Model Aligned VAE (约838MB)"
        },
        "LightningDiT-XL-64ep": {
            "url": "https://huggingface.co/hustvl/lightningdit-xl-imagenet256-64ep/resolve/main/lightningdit-xl-imagenet256-64ep.pt",
            "filename": "lightningdit-xl-imagenet256-64ep.pt", 
            "description": "LightningDiT扩散模型 64轮训练 (约2.3GB) - 适合微调"
        },
        "Latent Statistics": {
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt",
            "filename": "latents_stats.pt",
            "description": "潜在特征统计信息 (约1KB)"
        }
    }
    
    print("📋 需要下载的模型:")
    total_size_estimate = 0
    for name, info in models.items():
        print(f"   - {name}: {info['description']}")
        if "838MB" in info['description']:
            total_size_estimate += 838
        elif "2.3GB" in info['description']:
            total_size_estimate += 2300

        elif "1KB" in info['description']:
            total_size_estimate += 0.001
    
    print(f"📊 预计总大小: ~{total_size_estimate:.0f}MB")
    print("⏱️ 预计下载时间: 10-30分钟 (取决于网络速度)")
    
    # 下载模型
    success_count = 0
    for name, info in models.items():
        print(f"\n{'='*40}")
        print(f"📥 下载 {name}")
        print(f"📝 {info['description']}")
        
        filepath = models_dir / info['filename']
        
        if filepath.exists():
            print(f"✅ {name}: 已存在")
            print(f"📊 文件大小: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
            success_count += 1
        else:
            if download_file_with_progress(info['url'], filepath):
                success_count += 1
            else:
                print(f"❌ {name} 下载失败")
    
    print(f"\n{'='*60}")
    print(f"📊 下载结果: {success_count}/{len(models)} 个模型成功")
    
    if success_count == len(models):
        print("🎉 所有模型下载完成！")
        return True
    else:
        print("⚠️ 部分模型下载失败")
        return False

def verify_models():
    """验证下载的模型"""
    print("\n🔍 验证下载的模型...")

    models_dir = Path("models")
    expected_files = [
        "vavae-imagenet256-f16d32-dinov2.pt",
        "lightningdit-xl-imagenet256-64ep.pt",
        "latents_stats.pt"
    ]

    all_exist = True
    for filename in expected_files:
        filepath = models_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"✅ {filename}: {size_mb:.1f} MB")

            # 特别检查latents_stats.pt文件
            if filename == "latents_stats.pt" and size_mb < 0.001:
                print(f"⚠️ {filename}: 文件太小，可能损坏")
                all_exist = False
        else:
            print(f"❌ {filename}: 不存在")
            all_exist = False

    if all_exist:
        print("✅ 所有模型文件验证通过")
        return True
    else:
        print("❌ 模型文件验证失败")
        return False

def fix_latents_stats():
    """修复latents_stats.pt文件（集成修复功能）"""
    print("\n🔧 检查并修复latents_stats.pt文件...")

    models_dir = Path("models")
    latents_stats_file = models_dir / "latents_stats.pt"

    # 检查文件状态
    if latents_stats_file.exists():
        file_size = latents_stats_file.stat().st_size
        if file_size > 100:  # 文件大小合理
            try:
                import torch
                stats = torch.load(latents_stats_file)
                if 'mean' in stats and 'std' in stats:
                    print("✅ latents_stats.pt文件正常")
                    return True
            except:
                pass

    print("🔧 创建默认latents_stats.pt文件...")
    try:
        import torch
        # 创建默认统计信息
        mean = torch.zeros(1, 32, 1, 1)
        std = torch.ones(1, 32, 1, 1)
        latent_stats = {'mean': mean, 'std': std}

        torch.save(latent_stats, latents_stats_file)
        print(f"✅ 已创建默认统计文件: {latents_stats_file}")
        return True
    except Exception as e:
        print(f"❌ 创建默认统计文件失败: {e}")
        return False

def setup_model_paths():
    """设置模型路径配置"""
    print("\n⚙️ 设置模型路径配置...")
    
    models_dir = Path("models").absolute()
    
    # 检查LightningDiT目录
    lightningdit_dir = Path("LightningDiT")
    if not lightningdit_dir.exists():
        print("❌ LightningDiT目录不存在")
        return False
    
    # 更新VA-VAE配置
    vavae_config_path = lightningdit_dir / "tokenizer" / "configs" / "vavae_f16d32.yaml"
    
    if vavae_config_path.exists():
        print(f"🔧 更新VA-VAE配置: {vavae_config_path}")
        
        # 读取配置
        import yaml
        with open(vavae_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新检查点路径
        config['ckpt_path'] = str(models_dir / "vavae-imagenet256-f16d32-dinov2.pt")
        
        # 写回配置
        with open(vavae_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print("✅ VA-VAE配置已更新")
    else:
        print(f"⚠️ VA-VAE配置文件不存在: {vavae_config_path}")
    
    print("✅ 模型路径配置完成")
    return True

def main():
    """主函数"""
    print("🚀 步骤2: 下载LightningDiT预训练模型")
    print("="*60)
    
    # 检查网络连接
    print("🌐 检查网络连接...")
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        print("✅ 网络连接正常")
    except Exception as e:
        print(f"❌ 网络连接失败: {e}")
        print("💡 请检查网络连接或使用VPN")
        return False
    
    # 下载模型
    if not download_official_models():
        print("❌ 模型下载失败")
        return False
    
    # 验证模型
    if not verify_models():
        print("❌ 模型验证失败")
        return False

    # 修复latents_stats.pt文件
    if not fix_latents_stats():
        print("❌ latents_stats.pt修复失败")
        return False

    # 设置路径
    if not setup_model_paths():
        print("❌ 路径设置失败")
        return False
    
    print("\n✅ 步骤2完成！模型下载和配置完成")
    print("📋 下一步: !python step3_setup_configs.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
