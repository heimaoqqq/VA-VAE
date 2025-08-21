#!/usr/bin/env python3
"""
深度诊断MSE损失过高的问题
检查latent数据质量、分布、模型配置等关键因素
"""

import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
import matplotlib.pyplot as plt
import seaborn as sns

def load_latent_stats(latent_dir):
    """加载latent统计信息"""
    stats_file = latent_dir / 'latents_stats.pt'
    if stats_file.exists():
        return torch.load(stats_file)
    return None

def analyze_latent_files(latent_dir):
    """分析latent文件的数据质量"""
    print(f"\n🔍 分析latent目录: {latent_dir}")
    
    # 查找所有safetensors文件
    safetensor_files = list(latent_dir.glob("*.safetensors"))
    print(f"找到 {len(safetensor_files)} 个latent文件")
    
    if not safetensor_files:
        print("❌ 没有找到latent文件")
        return None
    
    # 分析第一个文件的数据
    sample_file = safetensor_files[0]
    print(f"分析样本文件: {sample_file.name}")
    
    latent_data = []
    labels = []
    
    try:
        with safe_open(sample_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"文件包含 {len(keys)} 个样本")
            
            # 加载前10个样本进行分析
            for i, key in enumerate(keys[:10]):
                if key.startswith('latent_'):
                    try:
                        latent = f.get_tensor(key)
                        latent_data.append(latent)
                        print(f"  成功加载: {key}, shape: {latent.shape}")
                        
                        # 获取对应的label
                        label_key = key.replace('latent_', 'label_')
                        if label_key in keys:
                            label = f.get_tensor(label_key)
                            labels.append(label.item())
                    except Exception as e:
                        print(f"  ❌ 加载{key}失败: {e}")
                        continue
    except Exception as e:
        print(f"❌ 打开safetensor文件失败: {e}")
        return None
    
    if not latent_data:
        print("❌ 没有找到有效的latent数据")
        return None
        
    # 转换为tensor进行分析
    latents = torch.stack(latent_data)  # [N, C, H, W]
    print(f"Latent shape: {latents.shape}")
    
    # 统计分析
    stats = {
        'shape': latents.shape,
        'mean': latents.mean().item(),
        'std': latents.std().item(),
        'min': latents.min().item(),
        'max': latents.max().item(),
        'zeros_ratio': (latents == 0).float().mean().item(),
        'channel_means': latents.mean(dim=[0, 2, 3]).tolist(),
        'channel_stds': latents.std(dim=[0, 2, 3]).tolist(),
    }
    
    print("\n📊 Latent数据统计:")
    print(f"  形状: {stats['shape']}")
    print(f"  均值: {stats['mean']:.6f}")
    print(f"  标准差: {stats['std']:.6f}")
    print(f"  范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"  零值比例: {stats['zeros_ratio']:.4f}")
    
    # 检查异常值
    if abs(stats['mean']) > 2.0:
        print("⚠️  警告: 均值过大，可能需要归一化")
    if stats['std'] > 10.0:
        print("⚠️  警告: 标准差过大，数据分布异常")
    if stats['zeros_ratio'] > 0.1:
        print("⚠️  警告: 零值过多，可能编码质量有问题")
    
    # 通道分析
    print("\n📈 各通道统计:")
    for i in range(min(8, len(stats['channel_means']))):  # 只显示前8个通道
        print(f"  通道{i:2d}: 均值={stats['channel_means'][i]:7.4f}, 标准差={stats['channel_stds'][i]:7.4f}")
    
    return stats

def check_data_consistency(train_dir, val_dir):
    """检查训练集和验证集数据一致性"""
    print("\n🔄 检查数据一致性...")
    
    train_stats = analyze_latent_files(train_dir) if train_dir.exists() else None
    val_stats = analyze_latent_files(val_dir) if val_dir.exists() else None
    
    if train_stats and val_stats:
        # 比较分布差异
        mean_diff = abs(train_stats['mean'] - val_stats['mean'])
        std_diff = abs(train_stats['std'] - val_stats['std'])
        
        print(f"\n🆚 训练集 vs 验证集:")
        print(f"  均值差异: {mean_diff:.6f}")
        print(f"  标准差差异: {std_diff:.6f}")
        
        if mean_diff > 0.5:
            print("⚠️  警告: 训练集和验证集均值差异较大")
        if std_diff > 0.5:
            print("⚠️  警告: 训练集和验证集标准差差异较大")

def simulate_dit_forward(latent_shape, num_classes=31):
    """模拟DiT前向传播检查维度匹配"""
    print(f"\n🤖 模拟DiT模型检查 (输入形状: {latent_shape})")
    
    batch_size, channels, height, width = latent_shape
    
    # 检查输入维度是否合理
    if channels != 32:
        print(f"⚠️  警告: 通道数{channels}不是预期的32")
    if height != 16 or width != 16:
        print(f"⚠️  警告: 空间尺寸{height}x{width}不是预期的16x16")
    
    # 模拟噪声添加
    noise = torch.randn_like(torch.zeros(latent_shape))
    noisy_latent = torch.zeros(latent_shape) + noise
    
    print(f"  噪声均值: {noise.mean():.6f}")
    print(f"  噪声标准差: {noise.std():.6f}")
    print(f"  加噪后范围: [{noisy_latent.min():.4f}, {noisy_latent.max():.4f}]")
    
    # 检查是否会产生数值不稳定
    if noisy_latent.abs().max() > 100:
        print("⚠️  警告: 加噪后数值过大，可能导致训练不稳定")

def main():
    """主诊断流程"""
    print("🔍 MSE损失过高深度诊断")
    print("=" * 50)
    
    # Kaggle路径
    base_paths = [
        Path('/kaggle/working/latents_official/vavae_config_for_dit'),
        Path('/kaggle/input/microdoppler-latents/latents_official/vavae_config_for_dit'),
        Path('/kaggle/working/latents')
    ]
    
    found_path = None
    for base_path in base_paths:
        if base_path.exists():
            found_path = base_path
            break
    
    if not found_path:
        print("❌ 未找到latent数据目录")
        return
    
    print(f"✅ 使用数据路径: {found_path}")
    
    # 检查训练集和验证集
    train_dir = found_path / 'microdoppler_train_256'
    val_dir = found_path / 'microdoppler_val_256'
    
    # 分析数据质量
    train_stats = None
    if train_dir.exists():
        train_stats = analyze_latent_files(train_dir)
    
    val_stats = None
    if val_dir.exists():
        val_stats = analyze_latent_files(val_dir)
    
    # 检查一致性
    check_data_consistency(train_dir, val_dir)
    
    # 模拟模型检查
    if train_stats:
        simulate_dit_forward(train_stats['shape'])
    
    # 生成诊断报告
    print("\n" + "="*50)
    print("🎯 MSE损失过高可能原因:")
    
    if train_stats:
        if abs(train_stats['mean']) > 1.0:
            print("❌ 数据未正确归一化 - 考虑启用latent_norm")
        if train_stats['std'] > 5.0:
            print("❌ 数据标准差过大 - 考虑调整latent_multiplier")
        if train_stats['zeros_ratio'] > 0.05:
            print("❌ 零值过多 - VA-VAE编码质量问题")
        
        if train_stats['shape'][1] != 32:
            print("❌ 通道数不匹配 - 检查VA-VAE配置")
        if train_stats['shape'][2] != 16 or train_stats['shape'][3] != 16:
            print("❌ 空间尺寸不匹配 - 检查下采样比例")
    
    print("\n🔧 建议的解决方案:")
    print("1. 如果数据均值不是0附近，启用latent_norm=true")
    print("2. 如果标准差过大，调整latent_multiplier")
    print("3. 如果通道/尺寸不匹配，重新编码数据")
    print("4. 考虑使用更大的学习率和更多epochs")
    print("5. 检查VA-VAE和DiT的配置兼容性")

if __name__ == "__main__":
    main()
