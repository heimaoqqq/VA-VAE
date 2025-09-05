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
import os
import glob
from torchvision import transforms


def load_real_dataset(data_dir="/kaggle/input/dataset", batch_size=8, num_samples=32):
    """加载真实的microdoppler数据集"""
    try:
        print(f"   尝试加载数据集: {data_dir}")
        
        # 检查数据集路径
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据集路径不存在: {data_dir}")
        
        # 扫描用户文件夹 (ID_1, ID_2, ..., ID_31)
        user_folders = []
        for i in range(1, 32):  # ID_1 到 ID_31
            user_folder = os.path.join(data_dir, f"ID_{i}")
            if os.path.exists(user_folder):
                user_folders.append((user_folder, i-1))  # 用户ID从0开始
        
        if not user_folders:
            raise FileNotFoundError("未找到任何用户文件夹 (ID_1 到 ID_31)")
        
        print(f"   发现 {len(user_folders)} 个用户文件夹")
        
        # 收集所有图像路径和用户ID
        all_image_paths = []
        all_user_ids = []
        
        for user_folder, user_id in user_folders:
            # 查找该用户的所有jpg图像
            jpg_files = glob.glob(os.path.join(user_folder, "*.jpg"))
            if jpg_files:
                all_image_paths.extend(jpg_files)
                all_user_ids.extend([user_id] * len(jpg_files))
        
        total_images = len(all_image_paths)
        print(f"   总图像数量: {total_images}")
        
        if total_images == 0:
            raise FileNotFoundError("未找到任何jpg图像")
        
        # 随机选择指定数量的样本
        indices = np.random.choice(total_images, min(num_samples, total_images), replace=False)
        selected_paths = [all_image_paths[i] for i in indices]
        selected_user_ids = [all_user_ids[i] for i in indices]
        
        # 定义图像变换
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # 不需要归一化，保持[0,1]范围
        ])
        
        # 加载图像
        images = []
        user_ids = []
        
        for img_path, user_id in zip(selected_paths, selected_user_ids):
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                user_ids.append(user_id)
            except Exception as e:
                print(f"   跳过损坏图像 {img_path}: {e}")
                continue
        
        if not images:
            raise ValueError("没有成功加载任何图像")
        
        images = torch.stack(images)
        user_ids = torch.tensor(user_ids)
        
        print(f"   成功加载样本数: {len(images)}")
        print(f"   图像形状: {images.shape}")
        print(f"   图像范围: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"   用户ID范围: [{user_ids.min().item()}, {user_ids.max().item()}]")
        
        return images, user_ids
        
    except Exception as e:
        print(f"   ⚠️ 无法加载真实数据集: {e}")
        print(f"   回退到模拟数据...")
        return create_fallback_images(batch_size, num_samples)


def create_fallback_images(batch_size=8, num_samples=32):
    """创建回退的模拟数据（当真实数据不可用时）"""
    print("   创建microdoppler风格的模拟数据...")
    
    images = []
    user_ids = []
    
    for i in range(num_samples):
        # 创建类似microdoppler的图案：时频谱特征
        img = np.zeros((3, 256, 256), dtype=np.float32)
        
        # 模拟多普勒频移曲线
        time_steps = np.linspace(0, 2*np.pi, 256)
        for t_idx, t in enumerate(time_steps):
            # 不同的多普勒模式
            if i % 4 == 0:
                # 正弦波多普勒
                freq_shift = 64 + 32 * np.sin(3*t + i*0.5)
            elif i % 4 == 1:
                # 线性调频
                freq_shift = 32 + 64 * t / (2*np.pi)
            elif i % 4 == 2:
                # 复合多普勒
                freq_shift = 64 + 16 * np.sin(2*t) + 16 * np.cos(4*t + i*0.3)
            else:
                # 阶跃多普勒
                freq_shift = 48 + 32 * (1 if np.sin(t) > 0 else -1)
            
            freq_shift = int(np.clip(freq_shift, 0, 255))
            
            # 在时频图上添加能量
            intensity = 0.3 + 0.4 * np.exp(-((t - np.pi)**2) / 2)
            img[0, freq_shift-2:freq_shift+3, t_idx] = intensity  # R通道
            img[1, freq_shift-1:freq_shift+2, t_idx] = intensity * 0.8  # G通道  
            img[2, freq_shift:freq_shift+1, t_idx] = intensity * 0.6  # B通道
        
        # 添加噪声
        noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 1)
        
        images.append(img)
        user_ids.append(i % 31)  # 31个用户
    
    images = torch.tensor(np.stack(images))
    user_ids = torch.tensor(user_ids)
    
    return images, user_ids


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


def test_vae_reconstruction(vae_checkpoint, data_dir="/kaggle/input/dataset", output_dir="./vae_validation", device=None):
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
    
    # 加载真实数据
    print("🎯 加载microdoppler测试数据...")
    test_images, test_user_ids = load_real_dataset(data_dir=data_dir, batch_size=8, num_samples=16)
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


def test_distribution_alignment(vae_checkpoint, data_dir="/kaggle/input/dataset", output_dir="./vae_validation", device=None):
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
        vae=vae,
        num_users=31,
        prototype_dim=768,
        enable_alignment=True,
        track_statistics=True
    ).to(device)
    
    # 加载真实数据进行测试
    test_images, test_user_ids = load_real_dataset(data_dir=data_dir, batch_size=16, num_samples=32)
    test_images = test_images.to(device)
    
    with torch.no_grad():
        # 获取VAE latents
        original_latents = vae.encode(test_images)
        
        # 更新分布统计
        alignment_model.update_statistics(original_latents)
        
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
    parser.add_argument('--data_dir', type=str, 
                      default='/kaggle/input/dataset',
                      help='MicroDoppler数据集路径')
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
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # 测试分布对齐
        alignment_enabled = test_distribution_alignment(
            vae_checkpoint=args.vae_checkpoint,
            data_dir=args.data_dir,
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
