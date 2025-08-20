#!/usr/bin/env python3
"""
诊断训练损失高的问题
检查数据归一化流程是否正确
"""

import torch
import yaml
import numpy as np
from pathlib import Path
import safetensors.torch

def diagnose_training():
    """诊断训练时的数据流和归一化"""
    print("=" * 50)
    print("训练损失诊断工具")
    print("=" * 50)
    
    # 1. 读取配置
    config_path = Path(__file__).parent / "config_dit_base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    latent_norm = config['data']['latent_norm']
    latent_multiplier = config['data']['latent_multiplier']
    
    print(f"\n📋 配置:")
    print(f"  latent_norm: {latent_norm}")
    print(f"  latent_multiplier: {latent_multiplier}")
    
    # 2. 检查原始latent数据
    data_path = Path("/kaggle/working/latent_dataset/train")
    if not data_path.exists():
        data_path = Path("G:/VA-VAE/microdoppler_finetune/latent_dataset/train")
    
    if not data_path.exists():
        print(f"❌ 数据路径不存在: {data_path}")
        return
    
    # 3. 加载latent统计文件
    stats_file = data_path / 'latents_stats.pt'
    if stats_file.exists():
        stats = torch.load(stats_file)
        saved_mean = stats['mean']
        saved_std = stats['std']
        print(f"\n📊 保存的统计信息:")
        print(f"  mean shape: {saved_mean.shape}, mean值: {saved_mean.mean():.4f}")
        print(f"  std shape: {saved_std.shape}, mean值: {saved_std.mean():.4f}")
    else:
        print(f"⚠️ 统计文件不存在: {stats_file}")
    
    # 4. 分析实际latent文件
    latent_files = list(data_path.glob("*.safetensors"))[:3]
    print(f"\n🔍 分析前3个latent文件:")
    
    for i, file in enumerate(latent_files):
        data = safetensors.torch.load_file(str(file))
        latent = data['latents']
        
        print(f"\n文件 {i+1}: {file.name}")
        print(f"  原始数据: mean={latent.mean():.4f}, std={latent.std():.4f}")
        
        # 模拟训练时的数据处理
        if latent_norm and 'saved_mean' in locals():
            # 模拟训练时的归一化
            normalized = (latent - saved_mean) / (saved_std + 1e-8)
            print(f"  归一化后: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
            
            # 检查归一化是否接近N(0,1)
            if abs(normalized.mean()) > 0.1 or abs(normalized.std() - 1.0) > 0.2:
                print(f"  ⚠️ 归一化后不是N(0,1)!")
            
            # 模拟损失计算
            noise = torch.randn_like(normalized)
            mse = ((normalized - noise) ** 2).mean()
            print(f"  模拟MSE损失: {mse:.4f}")
        else:
            # 无归一化情况：直接乘multiplier
            scaled = latent * latent_multiplier
            print(f"  缩放后: mean={scaled.mean():.4f}, std={scaled.std():.4f}")
            
            # 模拟损失计算
            noise = torch.randn_like(scaled)
            mse = ((scaled - noise) ** 2).mean()
            print(f"  模拟MSE损失: {mse:.4f}")
    
    # 6. 诊断建议
    print("\n" + "=" * 50)
    print("🔧 诊断建议:")
    
    if latent_norm:
        print("\n当前使用归一化训练，检查:")
        print("1. latents_stats.pt文件是否正确计算")
        print("2. 归一化后数据是否接近N(0,1)")
        print("3. latent_multiplier=1.0对VA-VAE是否正确")
    else:
        print("\n当前未使用归一化，可能导致:")
        print("1. 训练损失偏高（latent std≈1.5）")
        print("2. 考虑启用latent_norm=true")
    
    print("\n💡 可能的解决方案:")
    print("1. 确认latents_stats.pt统计正确")
    print("2. 检查训练代码的归一化实现")
    print("3. 尝试不同的latent_multiplier值")
    print("4. 降低学习率到1e-5")

if __name__ == "__main__":
    diagnose_training()
