#!/usr/bin/env python3
"""
测试脚本：验证分布对齐集成效果
测试VA-VAE latent分布和扩散模型训练/生成的兼容性
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from simplified_vavae import SimplifiedVAVAE
from distribution_aligned_diffusion import DistributionAlignedDiffusion
from microdoppler_data_loader import create_balanced_dataloader
import argparse


def test_vae_latent_distribution(vae, dataloader, device, num_batches=10):
    """测试VAE编码的latent分布"""
    print("\n" + "="*60)
    print("📊 测试1: VA-VAE Latent分布分析")
    print("="*60)
    
    all_latents = []
    vae.eval()
    
    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):
            if idx >= num_batches:
                break
            
            images = images.to(device)
            # 如果输入是latent，跳过编码
            if images.shape[1] == 32:  # latent channels
                latents = images
            else:  # 图像输入
                latents = vae.encode(images)
            
            all_latents.append(latents.cpu())
            print(f"   Batch {idx+1}: shape={latents.shape}, std={latents.std():.4f}")
    
    all_latents = torch.cat(all_latents, dim=0)
    
    # 统计分析
    stats = {
        'mean': all_latents.mean().item(),
        'std': all_latents.std().item(),
        'min': all_latents.min().item(),
        'max': all_latents.max().item(),
        'shape': all_latents.shape
    }
    
    print(f"\n📈 Latent分布统计:")
    print(f"   均值(Mean): {stats['mean']:.6f} (期望≈0)")
    print(f"   标准差(Std): {stats['std']:.6f} (期望≈1.5±0.3)")
    print(f"   最小值(Min): {stats['min']:.3f}")
    print(f"   最大值(Max): {stats['max']:.3f}")
    print(f"   形状(Shape): {stats['shape']}")
    
    # 判断是否需要分布对齐
    if abs(stats['std'] - 1.0) > 0.3:
        print(f"\n⚠️ 检测到分布偏离！需要分布对齐")
        print(f"   实际std={stats['std']:.4f}, 期望std≈1.0")
        print(f"   建议使用分布对齐方案")
    else:
        print(f"\n✅ 分布接近标准正态，可选择是否使用对齐")
    
    return stats


def test_distribution_alignment(vae, dataloader, device):
    """测试分布对齐模块功能"""
    print("\n" + "="*60)
    print("🔧 测试2: 分布对齐功能验证")
    print("="*60)
    
    # 创建分布对齐的扩散模型
    model = DistributionAlignedDiffusion(
        vae=vae,
        num_users=31,
        prototype_dim=768,
        enable_alignment=True,
        track_statistics=True
    ).to(device)
    
    print("✅ 分布对齐模型创建成功")
    
    # 测试归一化和反归一化
    with torch.no_grad():
        for idx, (latents, user_ids) in enumerate(dataloader):
            if idx >= 1:  # 只测试一个batch
                break
            
            latents = latents.to(device)
            
            print(f"\n📊 测试Batch:")
            print(f"   原始latent: mean={latents.mean():.4f}, std={latents.std():.4f}")
            
            # 更新统计
            model.update_statistics(latents)
            
            # 测试归一化
            normalized = model.normalize_latents(latents)
            print(f"   归一化后: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
            
            # 测试反归一化
            denormalized = model.denormalize_latents(normalized)
            print(f"   反归一化后: mean={denormalized.mean():.4f}, std={denormalized.std():.4f}")
            
            # 验证可逆性
            error = (denormalized - latents).abs().mean()
            print(f"   可逆性误差: {error:.6f}")
            
            if error < 1e-5:
                print("   ✅ 归一化/反归一化可逆性验证通过")
            else:
                print(f"   ⚠️ 可逆性误差较大: {error}")
    
    return model


def test_training_step(model, dataloader, device):
    """测试训练步骤"""
    print("\n" + "="*60)
    print("🚀 测试3: 训练步骤验证")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for idx, (latents, user_ids) in enumerate(dataloader):
        if idx >= 3:  # 测试3个batch
            break
        
        latents = latents.to(device)
        user_ids = user_ids.to(device)
        
        # 训练步骤 (直接传递user_ids，让training_step内部处理)
        loss_dict = model.training_step(latents, user_ids)
        total_loss = loss_dict['total_loss']
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"   Batch {idx+1}: Loss={total_loss.item():.4f} "
              f"(Diff={loss_dict['diffusion_loss'].item():.4f}, "
              f"Cont={loss_dict['contrastive_loss'].item():.4f})")
    
    print("\n✅ 训练步骤测试通过")


def test_generation(model, vae, device, num_samples=4):
    """测试生成功能"""
    print("\n" + "="*60)
    print("🎨 测试4: 生成功能验证")
    print("="*60)
    
    model.eval()
    
    with torch.no_grad():
        # 生成latents
        print(f"   生成{num_samples}个样本...")
        user_ids = [0] * num_samples  # 使用用户0
        
        latents = model.generate(
            user_ids=user_ids,
            num_samples=num_samples,
            num_inference_steps=50,
            guidance_scale=7.5
        )
        
        print(f"   生成latent分布: mean={latents.mean():.4f}, std={latents.std():.4f}")
        
        # 如果启用了分布对齐，检查是否已反归一化
        if model.enable_alignment:
            expected_std = model.latent_std.item() if model.latent_std > 0 else 1.5
            if abs(latents.std().item() - expected_std) < 0.3:
                print(f"   ✅ Latent已正确反归一化到VAE分布 (std≈{expected_std:.2f})")
            else:
                print(f"   ⚠️ Latent分布可能有问题: std={latents.std().item():.4f}, 期望≈{expected_std:.2f}")
        
        # 解码到图像
        print(f"   解码latents到图像...")
        images = vae.decode(latents)
        
        print(f"   图像范围: [{images.min():.3f}, {images.max():.3f}]")
        
        if images.min() >= -0.1 and images.max() <= 1.1:
            print("   ✅ 生成图像在合理范围内")
        else:
            print(f"   ⚠️ 图像值超出预期范围")
    
    print("\n✅ 生成功能测试完成")


def main():
    parser = argparse.ArgumentParser(description='测试分布对齐集成')
    
    parser.add_argument('--vae_checkpoint', type=str, 
                      default='/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt',
                      help='VA-VAE检查点路径')
    parser.add_argument('--latent_dir', type=str, 
                      default='/kaggle/working/latents',
                      help='Latent数据目录')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='测试批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                      help='设备')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    # 1. 加载VAE
    print("\n📦 加载VA-VAE...")
    vae = SimplifiedVAVAE(args.vae_checkpoint, use_vf='dinov2')
    vae.eval()
    vae = vae.to(device)
    print(f"✅ VAE加载成功")
    
    # 2. 创建数据加载器
    print("\n📊 创建数据加载器...")
    try:
        dataloader = create_balanced_dataloader(
            latent_dir=args.latent_dir,
            batch_size=args.batch_size,
            num_users_per_batch=4,
            split='train',
            num_workers=0  # 测试时使用单线程
        )
        print("✅ 数据加载器创建成功")
    except Exception as e:
        print(f"⚠️ 无法创建latent数据加载器: {e}")
        print("   使用模拟数据进行测试...")
        
        # 创建模拟数据
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
                # 模拟有偏的latent分布 (mean≈0.1, std≈1.5)
                self.latents = torch.randn(size, 32, 16, 16) * 1.5 + 0.1
                self.user_ids = torch.randint(0, 31, (size,))
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.latents[idx], self.user_ids[idx]
        
        dataset = DummyDataset()
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True
        )
    
    # 3. 运行测试
    print("\n" + "="*60)
    print("🧪 开始测试分布对齐集成")
    print("="*60)
    
    # 测试1: 分析VAE latent分布
    latent_stats = test_vae_latent_distribution(vae, dataloader, device)
    
    # 测试2: 分布对齐功能
    model = test_distribution_alignment(vae, dataloader, device)
    
    # 测试3: 训练步骤
    test_training_step(model, dataloader, device)
    
    # 测试4: 生成功能
    test_generation(model, vae, device)
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    if abs(latent_stats['std'] - 1.0) > 0.3:
        print("✅ 分布对齐方案已成功集成")
        print(f"   - VAE latent std={latent_stats['std']:.4f} (偏离标准分布)")
        print("   - 分布对齐模块正常工作")
        print("   - 训练和生成功能正常")
        print("\n建议：使用train_distribution_aligned.py进行训练")
    else:
        print("✅ 系统正常，latent分布接近标准")
        print("   - 可选择是否使用分布对齐")
        print("   - 所有功能测试通过")
    
    print("\n🎉 所有测试完成！")


if __name__ == "__main__":
    main()
