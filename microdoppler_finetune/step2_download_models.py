#!/usr/bin/env python3
"""
步骤2: 下载VA-VAE预训练模型
下载微调所需的预训练权重
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

def download_file(url: str, dest_path: Path, expected_size: int = None):
    """下载文件并显示进度条"""
    # 检查文件是否已存在
    if dest_path.exists():
        print(f"✅ 文件已存在: {dest_path.name}")
        return True
    
    print(f"📥 下载: {dest_path.name}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ 下载完成: {dest_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if dest_path.exists():
            dest_path.unlink()  # 删除不完整的文件
        return False

def verify_model_checksum(file_path: Path, expected_hash: str = None):
    """验证模型文件的完整性"""
    if not file_path.exists():
        return False
    
    if expected_hash:
        print(f"🔍 验证文件完整性: {file_path.name}")
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        if sha256_hash.hexdigest()[:8] == expected_hash[:8]:
            print(f"✅ 文件验证通过")
            return True
        else:
            print(f"❌ 文件验证失败")
            return False
    
    # 基本大小检查
    file_size = file_path.stat().st_size
    if file_size < 1000:  # 小于1KB，可能是错误文件
        print(f"⚠️ 文件大小异常: {file_size} bytes")
        return False
    
    return True

def download_vavae_models():
    """下载VA-VAE预训练模型"""
    print("📥 下载VA-VAE和LightningDiT预训练模型")
    print("="*60)
    
    # 检测环境
    if os.path.exists('/kaggle/working'):
        base_path = Path('/kaggle/working/VA-VAE')
        print("📍 Kaggle环境检测")
    else:
        base_path = Path.cwd()
        print("📍 本地环境检测")
    
    # 创建LightningDiT模型目录
    lightningdit_models_dir = base_path / "LightningDiT" / "models"
    lightningdit_models_dir.mkdir(parents=True, exist_ok=True)
    
    # 只需要LightningDiT模型（VA-VAE使用微调后的）
    models = {
        "LightningDiT B": {
            "url": "https://huggingface.co/hustvl/lightningdit-b-imagenet256-64ep/resolve/main/lightningdit-b-imagenet256-64ep.pt",
            "filename": "lightningdit-b-imagenet256-64ep.pt",
            "size_mb": 2800,  # B模型约2.8GB
            "description": "LightningDiT-B预训练权重 (ImageNet 256x256)",
            "required": True,
            "dest_dir": lightningdit_models_dir
        },
        "Latent Statistics": {
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt",
            "filename": "latents_stats.pt",
            "size_mb": 0.001,
            "description": "潜在空间统计信息（用于采样）",
            "required": False,
            "dest_dir": lightningdit_models_dir
        }
    }
    
    # 如果需要从checkpoint恢复，添加额外选项
    if '--resume' in sys.argv:
        models["Stage1 Checkpoint"] = {
            "url": None,  # 从之前的训练获取
            "filename": "vavae_stage1_checkpoint.pt",
            "size_mb": 2050,
            "description": "第一阶段训练检查点",
            "required": False
        }
    
    print("\n📋 模型列表:")
    total_size = 0
    for name, info in models.items():
        status = "必需" if info['required'] else "可选"
        size_display = f"{info['size_mb'] / 1024:.1f} GB" if info['size_mb'] > 1024 else f"{info['size_mb']} MB"
        print(f"   {name}: {info['description']} ({size_display}) [{status}]")
        if info['required']:
            total_size += info['size_mb']
    
    total_size_gb = total_size / 1024 if total_size > 1024 else 0
    if total_size_gb > 1:
        print(f"\n💾 总下载大小: ~{total_size_gb:.1f} GB")
    else:
        print(f"\n💾 总下载大小: ~{total_size} MB")
    
    # 下载模型
    success_count = 0
    failed_models = []
    
    for name, info in models.items():
        if info['url'] is None:
            print(f"\n⏭️ 跳过 {name} (需要手动提供)")
            continue
        
        # 使用每个模型指定的目标目录
        dest_dir = info.get('dest_dir', lightningdit_models_dir)
        dest_path = dest_dir / info['filename']
        
        print(f"\n📦 处理: {name}")
        print(f"   目标路径: {dest_path}")
        if download_file(info['url'], dest_path):
            if verify_model_checksum(dest_path):
                success_count += 1
            else:
                failed_models.append(name)
                if info['required']:
                    print(f"❌ 必需模型 {name} 验证失败!")
        else:
            failed_models.append(name)
            if info['required']:
                print(f"❌ 必需模型 {name} 下载失败!")
    
    # 创建模型配置文件
    print("\n📝 创建模型配置...")
    model_config = {
        "vavae_checkpoint": str(lightningdit_models_dir / "vavae-ema.pt"),
        "lightningdit_checkpoint": str(lightningdit_models_dir / "lightningdit-xl-imagenet256-64ep.pt"),
        "latent_stats": str(lightningdit_models_dir / "latents_stats.pt") if (lightningdit_models_dir / "latents_stats.pt").exists() else None,
        "model_type": "VA-VAE",
        "latent_dim": 32,
        "vfm_type": "dinov2",
        "input_size": 256
    }
    
    import json
    config_path = base_path / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✅ 配置已保存到: {config_path}")
    
    # 总结
    print("\n" + "="*60)
    print("📊 下载总结:")
    print(f"   成功: {success_count}/{len([m for m in models.values() if m['url']])}")
    if failed_models:
        print(f"   失败: {', '.join(failed_models)}")
        print("\n⚠️ 部分模型下载失败，请检查网络连接后重试")
    else:
        print("\n✅ 所有模型下载成功!")
    
    print("\n下一步:")
    print("1. 运行 python step3_prepare_dataset.py 准备数据集")
    print("2. 运行 python step4_train_stage1.py 开始第一阶段训练")
    
    return lightningdit_models_dir

def check_kaggle_models():
    """检查Kaggle输入目录中的预训练模型"""
    print("\n🔍 检查Kaggle输入目录...")
    
    kaggle_inputs = [
        "/kaggle/input/vavae-pretrained",
        "/kaggle/input/va-vae-models",
        "/kaggle/input/lightningdit-models"
    ]
    
    found_models = []
    for input_dir in kaggle_inputs:
        if os.path.exists(input_dir):
            print(f"✅ 找到输入目录: {input_dir}")
            for file in Path(input_dir).glob("*.pt"):
                print(f"   - {file.name} ({file.stat().st_size / 1024**2:.1f} MB)")
                found_models.append(str(file))
    
    if found_models:
        print(f"\n✅ 在Kaggle输入中找到 {len(found_models)} 个模型文件")
        print("   可以直接使用这些模型，无需下载")
        
        # 创建软链接
        models_dir = Path('/kaggle/working/models')
        models_dir.mkdir(exist_ok=True)
        
        for model_path in found_models:
            model_file = Path(model_path)
            link_path = models_dir / model_file.name
            if not link_path.exists():
                os.symlink(model_path, link_path)
                print(f"   链接: {model_file.name}")
        
        return True
    
    return False

if __name__ == "__main__":
    # Kaggle环境优先检查输入目录
    if os.path.exists('/kaggle/working'):
        if check_kaggle_models():
            print("\n✅ 使用Kaggle输入中的模型")
        else:
            print("\n⚠️ Kaggle输入中未找到模型，开始下载...")
            download_vavae_models()
    else:
        # 本地环境直接下载
        download_vavae_models()
