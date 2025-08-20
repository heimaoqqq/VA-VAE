#!/usr/bin/env python3
"""
验证VA-VAE配置正确性
检查latent normalization和multiplier设置是否正确
"""

import yaml
import torch
import numpy as np
from pathlib import Path
import safetensors.torch

def verify_config():
    """验证配置文件和latent数据的一致性"""
    print("=" * 50)
    print("VA-VAE配置验证工具")
    print("=" * 50)
    
    # 1. 读取配置文件
    config_path = Path(__file__).parent / "config_dit_base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n📋 当前配置:")
    print(f"  latent_norm: {config['data']['latent_norm']}")
    print(f"  latent_multiplier: {config['data']['latent_multiplier']}")
    
    # 2. 分析实际latent数据
    latent_path = Path("/kaggle/working/latent_dataset/train")
    if not latent_path.exists():
        print(f"\n⚠️ 警告: Kaggle路径不存在，尝试本地路径")
        latent_path = Path("G:/VA-VAE/microdoppler_finetune/latent_dataset/train")
    
    if latent_path.exists():
        print(f"\n📊 分析latent数据: {latent_path}")
        
        # 读取几个latent文件
        latent_files = list(latent_path.glob("*.safetensors"))[:5]
        if latent_files:
            all_means = []
            all_stds = []
            
            for file in latent_files:
                data = safetensors.torch.load_file(str(file))
                # 检查实际的键名
                keys = list(data.keys())
                print(f"  文件键名: {keys}")
                
                # 通常键名可能是'latents'或第一个键
                if 'latents' in data:
                    latent = data['latents']
                elif len(keys) > 0:
                    latent = data[keys[0]]  # 使用第一个键
                else:
                    continue
                    
                all_means.append(latent.mean().item())
                all_stds.append(latent.std().item())
            
            avg_mean = np.mean(all_means)
            avg_std = np.mean(all_stds)
            
            print(f"  平均mean: {avg_mean:.4f}")
            print(f"  平均std: {avg_std:.4f}")
            
            # 判断是否被缩放
            is_scaled = abs(avg_std - 0.18215) < 0.05
            is_unscaled = abs(avg_std - 1.5) < 0.5
            
            print(f"\n🔍 数据分析:")
            if is_scaled:
                print("  ❌ 数据似乎已被0.18215缩放（类似SD-VAE）")
                print("  建议: latent_multiplier应该设为1/0.18215≈5.5")
            elif is_unscaled:
                print("  ✅ 数据未被缩放（原始VA-VAE latent）")
                print("  建议: latent_multiplier应该设为1.0")
            else:
                print(f"  ⚠️ 数据标准差异常: {avg_std:.4f}")
    
    # 3. 验证配置合理性
    print("\n🎯 配置验证结果:")
    
    # VA-VAE应该使用1.0的multiplier
    if config['data']['latent_multiplier'] == 1.0:
        print("  ✅ latent_multiplier=1.0 正确（VA-VAE不需要缩放）")
    else:
        print(f"  ⚠️ latent_multiplier={config['data']['latent_multiplier']} 可能不正确")
        print("     VA-VAE通常使用1.0，SD-VAE使用0.18215")
    
    # 归一化通常应该开启以降低训练损失
    if config['data']['latent_norm']:
        print("  ✅ latent_norm=true 正确（训练时归一化可降低损失）")
    else:
        print("  ⚠️ latent_norm=false 可能导致训练损失偏高")
    
    # 4. 检查训练和生成的一致性
    print("\n📝 数据流程说明:")
    print("  1. 编码: 图像 → VA-VAE → 原始latents (std≈1.5)")
    print("  2. 训练: 加载latents → 归一化到N(0,1) → DiT训练")
    print("  3. 生成: DiT输出N(0,1) → 反归一化 → VA-VAE解码")
    
    print("\n✨ 关键点:")
    print("  • VA-VAE不使用SD-VAE的0.18215缩放")
    print("  • latent_norm=true降低训练损失")
    print("  • 生成时需正确反归一化")
    
    # 5. 检查生成代码
    train_script = Path(__file__).parent / "step8_train_dit_from_scratch.py"
    if train_script.exists():
        with open(train_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if "VA-VAE反归一化流程" in content:
                print("\n✅ 生成代码已更新VA-VAE注释")
            else:
                print("\n⚠️ 生成代码可能需要更新注释")

if __name__ == "__main__":
    verify_config()
