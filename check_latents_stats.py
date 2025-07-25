#!/usr/bin/env python3
"""
检查latents_stats.pt文件内容
验证统计信息是否正确
"""

import torch
import os
from pathlib import Path

def check_latents_stats():
    """检查latents_stats.pt文件"""
    
    print("🔍 检查latents_stats.pt文件")
    print("=" * 40)
    
    stats_file = Path("official_models/latents_stats.pt")
    
    if not stats_file.exists():
        print(f"❌ 文件不存在: {stats_file}")
        return False
    
    # 检查文件大小
    file_size = stats_file.stat().st_size
    print(f"📁 文件大小: {file_size} bytes ({file_size/1024:.1f} KB)")
    
    try:
        # 加载文件
        stats = torch.load(stats_file, map_location='cpu')
        print(f"✅ 文件加载成功")
        
        # 检查内容结构
        print(f"\n📋 文件内容结构:")
        if isinstance(stats, dict):
            print(f"   类型: 字典")
            print(f"   键: {list(stats.keys())}")
            
            # 检查mean和std
            if 'mean' in stats and 'std' in stats:
                mean = stats['mean']
                std = stats['std']
                
                print(f"\n📊 统计信息:")
                print(f"   mean shape: {mean.shape}")
                print(f"   mean dtype: {mean.dtype}")
                print(f"   mean range: [{mean.min():.4f}, {mean.max():.4f}]")
                
                print(f"   std shape: {std.shape}")
                print(f"   std dtype: {std.dtype}")
                print(f"   std range: [{std.min():.4f}, {std.max():.4f}]")
                
                # 检查维度是否正确 (应该是32维)
                if mean.shape[0] == 32 and std.shape[0] == 32:
                    print(f"✅ 维度正确: 32通道")
                else:
                    print(f"❌ 维度错误: 期望32，实际{mean.shape[0]}")
                
                return True
            else:
                print(f"❌ 缺少mean或std字段")
                return False
        else:
            print(f"❌ 文件格式错误: 期望字典，实际{type(stats)}")
            return False
            
    except Exception as e:
        print(f"❌ 文件加载失败: {e}")
        return False

def check_all_files():
    """检查所有模型文件"""
    
    print("\n📁 检查所有模型文件")
    print("=" * 40)
    
    models_dir = Path("official_models")
    required_files = [
        ("vavae-imagenet256-f16d32-dinov2.pt", "VA-VAE模型"),
        ("lightningdit-xl-imagenet256-800ep.pt", "LightningDiT模型"),
        ("latents_stats.pt", "潜在特征统计")
    ]
    
    all_exist = True
    for filename, description in required_files:
        filepath = models_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024*1024)
            print(f"✅ {description}: {size_mb:.1f} MB")
        else:
            print(f"❌ {description}: 文件不存在")
            all_exist = False
    
    return all_exist

def main():
    """主函数"""
    print("🔍 检查预下载的模型文件")
    print("=" * 50)
    
    # 检查所有文件
    if not check_all_files():
        print("\n❌ 部分文件缺失")
        return
    
    # 详细检查latents_stats.pt
    if check_latents_stats():
        print("\n✅ latents_stats.pt文件正确！")
        print("📝 这个文件包含了ImageNet-256数据集上VA-VAE编码后的")
        print("   潜在特征的均值和标准差统计信息，用于归一化。")
    else:
        print("\n❌ latents_stats.pt文件有问题")

if __name__ == "__main__":
    main()
