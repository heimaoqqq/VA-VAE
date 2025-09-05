#!/usr/bin/env python3
"""
VAE重建验证脚本
测试VA-VAE的编码/解码功能和分布对齐的效果
这比未训练的扩散样本更有意义
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from simplified_vavae import SimplifiedVAVAE
from distribution_aligned_diffusion import DistributionAlignedDiffusion
import argparse


def create_sample_images(batch_size=8, image_size=256, channels=3):
    """创建示例图像用于测试"""
    # 创建一些有意义的测试图像
    images = []
    
    for i in range(batch_size):
        # 创建不同的几何图案
        img = np.zeros((channels, image_size, image_size), dtype=np.float32)
        
        if i % 4 == 0:
            # 渐变图案
            for c in range(channels):
                img[c] = np.linspace(0, 1, image_size).reshape(1, -1)
        elif i % 4 == 1:
            # 棋盘图案
            for y in range(0, image_size, 32):
                for x in range(0, image_size, 32):
                    if (y//32 + x//32) % 2 == 0:
                        img[:, y:y+32, x:x+32] = 0.8
        elif i % 4 == 2:
            # 圆形图案
            center = image_size // 2
            y, x = np.ogrid[:image_size, :image_size]
            mask = (x - center)**2 + (y - center)**2 < (image_size//4)**2
            img[:, mask] = 0.7
        else:
            # 噪声图案
            img = np.random.rand(channels, image_size, image_size).astype(np.float32) * 0.5 + 0.25
        
        images.append(img)
    
    return torch.tensor(np.stack(images))


def save_comparison_grid(original, reconstructed, filepath, titles=None):
    """保存原图与重建图的对比网格"""
    batch_size = original.shape[0]
    
    # 确保图像在[0,1]范围内
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # 转换为numpy
    orig_np = original.cpu().numpy().transpose(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
    recon_np = reconstructed.cpu().numpy().transpose(0, 2, 3, 1)
    
    # 创建对比网格 (2行: 原图 + 重建)
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size*3, 6))
    
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(batch_size):
        # 原图
        ax_orig = axes[0, i] if batch_size > 1 else axes[0]
        if orig_np.shape[-1] == 3:  # RGB
            ax_orig.imshow(orig_np[i])
        else:  # 灰度
            ax_orig.imshow(orig_np[i, :, :, 0], cmap='gray')
        ax_orig.axis('off')
        ax_orig.set_title(f'Original {i+1}' if titles is None else f'Orig: {titles[i]}')
        
        # 重建图
        ax_recon = axes[1, i] if batch_size > 1 else axes[1]
        if recon_np.shape[-1] == 3:  # RGB
            ax_recon.imshow(recon_np[i])
        else:  # 灰度
            ax_recon.imshow(recon_np[i, :, :, 0], cmap='gray')
        ax_recon.axis('off')
        ax_recon.set_title(f'Reconstructed {i+1}')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 对比图已保存到: {filepath}")


def analyze_latent_distribution(latents, title="Latent Distribution"):
    """分析latent分布并可视化"""
    print(f"\n📊 {title}")
    print("-" * 50)
    
    # 基本统计
    mean = latents.mean().item()
    std = latents.std().item()
    min_val = latents.min().item()
    max_val = latents.max().item()
    
    print(f"   均值 (Mean): {mean:.4f}")
    print(f"   标准差 (Std): {std:.4f}")
    print(f"   最小值 (Min): {min_val:.4f}")
    print(f"   最大值 (Max): {max_val:.4f}")
    print(f"   形状 (Shape): {latents.shape}")
    
    # 检查分布特性
    if abs(mean) < 0.1:
        print("   ✅ 均值接近0")
    else:
        print(f"   ⚠️ 均值偏离0: {mean:.4f}")
    
    if 0.8 < std < 2.0:
        print("   ✅ 标准差在合理范围")
    else:
        print(f"   ⚠️ 标准差异常: {std:.4f}")
    
    return {"mean": mean, "std": std, "min": min_val, "max": max_val}


def test_vae_reconstruction(vae_checkpoint, output_dir="./vae_validation", device=None):
    """测试VAE重建功能"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🎨 使用设备: {device}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 加载VAE
    print("📦 加载VA-VAE...")
    vae = SimplifiedVAVAE(checkpoint_path=vae_checkpoint)
    vae = vae.to(device)
    vae.eval()
    
    # 创建测试图像
    print("🎯 创建测试图像...")
    test_images = create_sample_images(batch_size=8, image_size=256, channels=3)
    test_images = test_images.to(device)
    
    print(f"   测试图像形状: {test_images.shape}")
    print(f"   图像范围: [{test_images.min().item():.3f}, {test_images.max().item():.3f}]")
    
    # VAE编码解码测试
    print("\n🔄 执行VAE编码-解码测试...")
    
    with torch.no_grad():
        # 编码到latent
        latents = vae.encode(test_images)
        print(f"   编码latent形状: {latents.shape}")
        
        # 分析latent分布
        latent_stats = analyze_latent_distribution(latents, "VAE编码的Latent分布")
        
        # 解码回图像
        reconstructed = vae.decode(latents)
        print(f"   重建图像形状: {reconstructed.shape}")
        print(f"   重建图像范围: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
        
        # 计算重建误差
        mse_error = torch.nn.functional.mse_loss(test_images, reconstructed).item()
        mae_error = torch.nn.functional.l1_loss(test_images, reconstructed).item()
        
        print(f"\n📏 重建误差:")
        print(f"   MSE: {mse_error:.6f}")
        print(f"   MAE: {mae_error:.6f}")
        
        if mse_error < 0.01:
            print("   ✅ 重建质量优秀")
        elif mse_error < 0.05:
            print("   ✅ 重建质量良好")
        else:
            print("   ⚠️ 重建误差较大")
    
    # 保存对比图
    comparison_path = output_path / "vae_reconstruction_comparison.png"
    save_comparison_grid(test_images, reconstructed, comparison_path)
    
    return latent_stats, mse_error, mae_error


def test_distribution_alignment(vae_checkpoint, output_dir="./vae_validation", device=None):
    """测试分布对齐功能"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("🔧 测试分布对齐功能")
    print("="*60)
    
    # 加载VAE
    vae = SimplifiedVAVAE(checkpoint_path=vae_checkpoint)
    vae = vae.to(device)
    vae.eval()
    
    # 创建分布对齐模型（仅用于测试归一化功能）
    alignment_model = DistributionAlignedDiffusion(
        unet_config={
            'sample_size': (16, 16),
            'in_channels': 32,
            'out_channels': 32,
            'down_block_types': ['CrossAttnDownBlock2D'] * 3,
            'up_block_types': ['CrossAttnUpBlock2D'] * 3,
            'block_out_channels': [320, 640, 1280],
            'cross_attention_dim': 768,
            'layers_per_block': 2,
            'attention_head_dim': 8
        }
    ).to(device)
    
    # 创建测试图像并编码
    test_images = create_sample_images(batch_size=16, image_size=256, channels=3)
    test_images = test_images.to(device)
    
    with torch.no_grad():
        # 获取VAE latents
        original_latents = vae.encode(test_images)
        
        # 更新分布统计
        alignment_model._update_latent_stats(original_latents)
        
        print(f"🎯 检测到的latent分布:")
        print(f"   均值: {alignment_model.latent_mean.item():.4f}")
        print(f"   标准差: {alignment_model.latent_std.item():.4f}")
        print(f"   分布对齐状态: {'启用' if alignment_model.enable_alignment else '禁用'}")
        
        if alignment_model.enable_alignment:
            # 测试归一化
            normalized = alignment_model.normalize_latents(original_latents)
            norm_stats = analyze_latent_distribution(normalized, "归一化后的Latent分布")
            
            # 测试反归一化
            denormalized = alignment_model.denormalize_latents(normalized)
            
            # 检查可逆性
            error = torch.nn.functional.mse_loss(original_latents, denormalized).item()
            print(f"\n🔄 归一化可逆性测试:")
            print(f"   原始 -> 归一化 -> 反归一化 误差: {error:.8f}")
            
            if error < 1e-6:
                print("   ✅ 完美可逆")
            elif error < 1e-4:
                print("   ✅ 高精度可逆")
            else:
                print("   ⚠️ 可逆性误差较大")
            
            # 验证归一化后的分布
            if abs(norm_stats["mean"]) < 0.1 and abs(norm_stats["std"] - 1.0) < 0.1:
                print("   ✅ 归一化后分布接近N(0,1)")
            else:
                print("   ⚠️ 归一化后分布偏离N(0,1)")
        
        else:
            print("   ℹ️ latent分布接近标准分布，无需对齐")
    
    return alignment_model.enable_alignment


def main():
    parser = argparse.ArgumentParser(description='VAE重建和分布对齐验证')
    parser.add_argument('--vae_checkpoint', type=str, 
                      default='/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt',
                      help='VA-VAE检查点路径')
    parser.add_argument('--output_dir', type=str, default='./vae_validation',
                      help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                      help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("🔍 VA-VAE重建和分布对齐验证")
    print("="*60)
    
    try:
        # 测试VAE重建
        latent_stats, mse_error, mae_error = test_vae_reconstruction(
            vae_checkpoint=args.vae_checkpoint,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # 测试分布对齐
        alignment_enabled = test_distribution_alignment(
            vae_checkpoint=args.vae_checkpoint,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # 总结
        print("\n" + "="*60)
        print("📋 验证总结")
        print("="*60)
        print(f"✅ VAE重建功能: {'正常' if mse_error < 0.05 else '异常'}")
        print(f"✅ Latent分布: std={latent_stats['std']:.3f}")
        print(f"✅ 分布对齐: {'需要启用' if alignment_enabled else '无需启用'}")
        print(f"✅ 重建误差: MSE={mse_error:.6f}, MAE={mae_error:.6f}")
        
        if alignment_enabled:
            print(f"\n💡 建议: VAE latent分布偏离标准分布，训练时将自动启用分布对齐")
        else:
            print(f"\n💡 建议: VAE latent分布接近标准分布，可直接训练")
        
        print(f"\n📁 验证结果已保存到: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
