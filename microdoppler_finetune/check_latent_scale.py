"""
快速检查latent数据的缩放状态
"""
import torch
from safetensors.torch import load_file
from pathlib import Path
import numpy as np

def check_latent_scale():
    """检查latent是否已被0.18215缩放"""
    
    # 查找latent文件
    base_paths = [
        Path('/kaggle/working/latents_official/vavae_config_for_dit/microdoppler_train_256'),
        Path('/kaggle/input/microdoppler-latents/latents_official/vavae_config_for_dit/microdoppler_train_256'),
    ]
    
    data_dir = None
    for path in base_paths:
        if path.exists():
            data_dir = path
            break
    
    if not data_dir:
        print("❌ 找不到latent数据目录")
        return
    
    # 加载第一个文件
    files = list(data_dir.glob('*.safetensors'))
    if not files:
        print("❌ 没有找到safetensors文件")
        return
    
    print(f"✅ 检查文件: {files[0]}")
    data = load_file(files[0])
    
    # 获取latents
    if 'latents' in data:
        latents = data['latents']
    else:
        # 尝试其他可能的key
        for key in data.keys():
            if 'latent' in key.lower():
                latents = data[key]
                break
        else:
            print(f"❌ 找不到latents，可用keys: {list(data.keys())}")
            return
    
    print(f"\n📊 Latent统计:")
    print(f"  Shape: {latents.shape}")
    print(f"  Mean: {latents.mean():.6f}")
    print(f"  Std: {latents.std():.6f}")
    print(f"  Min: {latents.min():.6f}")
    print(f"  Max: {latents.max():.6f}")
    
    # 判断是否已缩放
    std = latents.std().item()
    mean = abs(latents.mean().item())
    
    print(f"\n🔍 缩放状态分析:")
    if std < 0.5 and mean < 0.1:
        print("  ✅ 数据已被0.18215缩放（std < 0.5）")
        print("  建议配置:")
        print("    latent_norm: false")
        print("    latent_multiplier: 1.0")
    elif std > 1.0:
        print("  ❌ 数据未缩放（std > 1.0）")
        print("  建议配置:")
        print("    latent_norm: false  # 不要归一化")
        print("    latent_multiplier: 0.18215  # 需要缩放")
    else:
        print("  ⚠️ 数据状态不明确，可能已部分处理")
        print("  当前std介于0.5-1.0之间")
    
    # 测试归一化后的效果
    print(f"\n🧪 归一化测试:")
    norm_latents = (latents - latents.mean()) / (latents.std() + 1e-8)
    print(f"  归一化后 Mean: {norm_latents.mean():.6f}")
    print(f"  归一化后 Std: {norm_latents.std():.6f}")
    
    # 测试0.18215缩放效果
    scaled_latents = latents * 0.18215
    print(f"\n🧪 缩放测试 (×0.18215):")
    print(f"  缩放后 Mean: {scaled_latents.mean():.6f}")
    print(f"  缩放后 Std: {scaled_latents.std():.6f}")

if __name__ == "__main__":
    check_latent_scale()
