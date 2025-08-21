"""快速检查latent统计信息"""
import torch
from safetensors import safe_open
import glob
import os

# 检查本地latent文件
latent_dir = "/kaggle/working/latents_official/vavae_config_for_dit/microdoppler_train_256"

if not os.path.exists(latent_dir):
    print(f"找不到目录: {latent_dir}")
    exit()

files = glob.glob(os.path.join(latent_dir, "*.safetensors"))
if not files:
    print("没有找到safetensors文件")
    exit()

print(f"找到 {len(files)} 个文件")
print(f"检查第一个文件: {files[0]}")

# 加载并分析
with safe_open(files[0], framework="pt", device="cpu") as f:
    latents = f.get_tensor("latents")
    print(f"\nLatent shape: {latents.shape}")
    print(f"Dtype: {latents.dtype}")
    
    # 计算统计
    mean = latents.mean().item()
    std = latents.std().item()
    min_val = latents.min().item()
    max_val = latents.max().item()
    
    print(f"\n📊 原始统计:")
    print(f"Mean: {mean:.6f}")
    print(f"Std:  {std:.6f}")
    print(f"Min:  {min_val:.6f}")
    print(f"Max:  {max_val:.6f}")
    
    # 测试不同缩放
    print(f"\n🔧 如果应用不同缩放:")
    print(f"× 0.18215: mean={mean*0.18215:.6f}, std={std*0.18215:.6f}")
    print(f"÷ 0.18215: mean={mean/0.18215:.6f}, std={std/0.18215:.6f}")
    
    # 判断状态
    print(f"\n💡 分析:")
    if abs(std - 0.18215) < 0.05:
        print("✅ Latent可能已被0.18215缩放")
        print("建议: latent_norm=false, latent_multiplier=5.5 (1/0.18215)")
    elif abs(std - 1.0) < 0.3:
        print("✅ Latent接近单位方差")
        print("建议: latent_norm=false, latent_multiplier=0.18215")
    else:
        print(f"⚠️ Latent方差={std:.2f}，不标准")
        print("建议: latent_norm=true, latent_multiplier=0.18215")
