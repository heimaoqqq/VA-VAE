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
            print(f"文件包含keys: {keys}")
            
            # 检查step6_encode_official.py的格式：latents, latents_flip, labels
            if 'latents' in keys:
                latents_tensor = f.get_tensor('latents')
                labels_tensor = f.get_tensor('labels') if 'labels' in keys else None
                
                print(f"  ✅ 找到latents tensor: {latents_tensor.shape}")
                if labels_tensor is not None:
                    print(f"  ✅ 找到labels tensor: {labels_tensor.shape}")
                
                # 分析前10个样本（如果有的话）
                num_samples = min(10, latents_tensor.shape[0])
                for i in range(num_samples):
                    latent_data.append(latents_tensor[i])
                    if labels_tensor is not None and i < labels_tensor.shape[0]:
                        labels.append(labels_tensor[i].item())
                
                print(f"  成功加载 {num_samples} 个样本进行分析")
            else:
                print(f"  ❌ 未找到预期的'latents' key，实际keys: {keys}")
                return None
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
    
    train_stats = None
    if train_dir.exists():
        print(f"📁 检查训练集: {train_dir}")
        train_stats = analyze_latent_files(train_dir)
    else:
        print(f"❌ 训练集目录不存在: {train_dir}")
    
    val_stats = None  
    if val_dir.exists():
        print(f"📁 检查验证集: {val_dir}")
        val_stats = analyze_latent_files(val_dir)
    else:
        print(f"❌ 验证集目录不存在: {val_dir}")
    
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

def simple_directory_check(base_path):
    """简单检查目录结构"""
    print(f"\n📁 检查目录结构: {base_path}")
    
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_path}")
        return
    
    # 列出所有子目录和文件
    items = list(base_path.iterdir())
    print(f"📋 找到 {len(items)} 个项目:")
    
    for item in sorted(items):
        if item.is_dir():
            # 检查子目录内容
            sub_files = list(item.glob("*.safetensors"))
            print(f"  📂 {item.name}/ - {len(sub_files)} 个safetensors文件")
            
            # 检查第一个文件的大小
            if sub_files:
                first_file = sub_files[0]
                size_mb = first_file.stat().st_size / (1024*1024)
                print(f"    📄 样本: {first_file.name} ({size_mb:.1f}MB)")
        else:
            size_mb = item.stat().st_size / (1024*1024)
            print(f"  📄 {item.name} ({size_mb:.1f}MB)")

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
        # 检查所有可能的父目录
        for base_path in base_paths:
            parent = base_path.parent
            if parent.exists():
                print(f"🔍 检查父目录: {parent}")
                simple_directory_check(parent)
        return
    
    print(f"✅ 使用数据路径: {found_path}")
    
    # 首先简单检查目录结构
    simple_directory_check(found_path)
    
    # 检查训练集和验证集
    train_dir = found_path / 'microdoppler_train_256'
    val_dir = found_path / 'microdoppler_val_256'
    
    print(f"\n🎯 预期目录:")
    print(f"  训练集: {train_dir} - {'✅存在' if train_dir.exists() else '❌不存在'}")
    print(f"  验证集: {val_dir} - {'✅存在' if val_dir.exists() else '❌不存在'}")
    
    # 只有目录存在时才进行详细分析
    train_stats = None
    val_stats = None
    try:
        # 检查一致性
        check_data_consistency(train_dir, val_dir)
        
        # 如果目录存在，获取统计信息用于后续分析
        if train_dir.exists():
            train_stats = analyze_latent_files(train_dir)
        if val_dir.exists():
            val_stats = analyze_latent_files(val_dir)
            
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 模拟模型检查
    if train_stats and train_stats.get('shape'):
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
