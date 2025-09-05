#!/usr/bin/env python3
"""
训练分布对齐的扩散模型 - 集成版本
解决VA-VAE与扩散模型之间的latent分布不匹配问题
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from microdoppler_data_loader import (
    MicroDopplerLatentDataset, 
    BalancedBatchSampler,
    prepare_latent_dataset,
    create_balanced_dataloader
)
from distribution_aligned_diffusion import DistributionAlignedDiffusion
from simplified_vavae import SimplifiedVAVAE

def compute_latent_statistics(dataloader, device, max_batches=50):
    """
    计算训练数据的latent分布统计
    """
    print(f"   🔍 分析前{max_batches}个批次的latent分布...")
    
    all_latents = []
    batch_count = 0
    
    for batch_idx, (latents, user_ids) in enumerate(dataloader):
        if batch_count >= max_batches:
            break
            
        latents = latents.to(device)
        all_latents.append(latents.cpu().flatten())
        batch_count += 1
        
        if batch_count % 10 == 0:
            print(f"   📊 已处理 {batch_count}/{max_batches} 批次...")
    
    # 合并所有latent数据
    all_latents = torch.cat(all_latents, dim=0)
    
    # 计算统计量
    latent_mean = all_latents.mean().item()
    latent_std = all_latents.std().item()
    latent_min = all_latents.min().item()
    latent_max = all_latents.max().item()
    
    print(f"   ✅ 统计完成：分析了 {len(all_latents):,} 个latent值")
    
    return {
        'mean': latent_mean,
        'std': latent_std, 
        'min': latent_min,
        'max': latent_max
    }

def train_distribution_aligned_diffusion(args):
    """训练分布对齐的扩散模型"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    sample_dir = output_dir / 'samples'
    sample_dir.mkdir(exist_ok=True)
    
    # 初始化VAE - 使用正确的VF配置
    print("\n📦 加载预训练VA-VAE...")
    vae_path = args.vae_checkpoint if hasattr(args, 'vae_checkpoint') else "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    vae = SimplifiedVAVAE(vae_path, use_vf='dinov2')  # 启用VF以匹配预训练模型
    vae.eval()
    vae.freeze()
    vae = vae.to(device)
    print(f"✅ VAE加载完成，缩放因子: {vae.scale_factor}")
    print(f"✅ VF模式: {'启用 (dinov2)' if vae.use_vf else '禁用'}")
    
    # 准备latent数据集
    latent_dir = Path(args.latent_dir)
    
    # 检查latent数据集是否完整
    def check_latent_dataset_complete(latent_dir, split_file):
        """检查latent数据集是否完整"""
        if not latent_dir.exists():
            return False, "目录不存在"
        
        if not (latent_dir / "train").exists() or not (latent_dir / "val").exists():
            return False, "缺少train或val子目录"
        
        # 检查是否有latent文件
        train_files = list((latent_dir / "train").glob("*.pt"))
        val_files = list((latent_dir / "val").glob("*.pt"))
        
        if len(train_files) == 0 or len(val_files) == 0:
            return False, f"latent文件不足: train={len(train_files)}, val={len(val_files)}"
        
        return True, f"数据集完整: train={len(train_files)}, val={len(val_files)} files"
    
    is_complete, status_msg = check_latent_dataset_complete(latent_dir, args.split_file)
    
    if not is_complete or args.prepare_latents:
        if args.prepare_latents:
            print("\n🔄 强制重新准备latent数据集...")
        else:
            print(f"\n🔄 数据集不完整({status_msg})，开始准备latent数据集...")
            
        latent_dir = prepare_latent_dataset(
            image_dir=args.image_dir,
            vae_model=vae,
            output_dir=args.latent_dir,
            split_file=args.split_file,
            device=device
        )
    else:
        print(f"\n✅ 发现完整的latent数据集: {status_msg}")
        print(f"   路径: {latent_dir}")
    
    # 创建数据加载器
    print("\n📊 创建数据加载器...")
    train_loader = create_balanced_dataloader(
        latent_dir=latent_dir,
        batch_size=args.batch_size,
        num_users_per_batch=args.num_users_per_batch,
        split='train',
        num_workers=args.num_workers
    )
    
    val_loader = create_balanced_dataloader(
        latent_dir=latent_dir,
        batch_size=args.batch_size,
        num_users_per_batch=args.num_users_per_batch,
        split='val',
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 计算训练数据的实际latent分布统计
    print("\n📊 计算训练数据latent分布统计...")
    latent_stats = compute_latent_statistics(train_loader, device, max_batches=50)
    
    print(f"📈 实际训练latent分布:")
    print(f"   Mean: {latent_stats['mean']:.6f}")
    print(f"   Std:  {latent_stats['std']:.6f}")
    print(f"   Range: [{latent_stats['min']:.2f}, {latent_stats['max']:.2f}]")
    print(f"   3σ Range: [{latent_stats['mean']-3*latent_stats['std']:.2f}, {latent_stats['mean']+3*latent_stats['std']:.2f}]")
    
    # 🔑 关键：检测是否需要分布对齐
    if abs(latent_stats['std'] - 1.0) > 0.2 or abs(latent_stats['mean']) > 0.1:
        print(f"\n⚠️ 检测到latent分布偏离标准正态分布!")
        print(f"   需要分布对齐：mean={latent_stats['mean']:.4f} (期望0), std={latent_stats['std']:.4f} (期望1)")
        use_distribution_alignment = True
    else:
        print(f"\n✅ Latent分布接近标准正态分布，可选择是否使用分布对齐")
        use_distribution_alignment = args.force_alignment
    
    # 创建分布对齐的扩散模型
    print(f"\n🚀 创建{'分布对齐' if use_distribution_alignment else '标准'}扩散模型...")
    model = DistributionAlignedDiffusion(
        vae=vae,
        num_users=args.num_users,
        prototype_dim=args.prototype_dim,
        enable_alignment=use_distribution_alignment,  # 根据检测结果启用对齐
        track_statistics=True  # 始终跟踪统计信息
    )
    
    # 将模型移动到设备
    model = model.to(device)
    print(f"✅ 模型已移动到设备: {device}")
    
    if use_distribution_alignment:
        print(f"📊 分布对齐已启用")
        print(f"   - 训练时将归一化latent到N(0,1)")
        print(f"   - 生成时将反归一化以匹配VAE分布")
    else:
        print(f"📊 分布对齐已禁用（latent已接近标准分布）")
    
    # 设置优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 使用余弦退火调度器，保留最小学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=1e-6  # 保持最小学习率
    )
    
    # 训练循环
    print("\n🎯 开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # 训练阶段
        model.train()
        train_losses = {
            'total': 0.0,
            'diffusion': 0.0, 
            'contrastive': 0.0
        }
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for batch_idx, (latents, user_ids) in enumerate(train_bar):
            latents = latents.to(device)
            user_ids = user_ids.to(device)
            
            # 第一个epoch的第一个batch显示详细信息
            if epoch == 0 and batch_idx == 0:
                print(f"\n📊 第一批训练数据分析:")
                print(f"   Shape: {latents.shape}")
                print(f"   原始分布: Mean={latents.mean():.6f}, Std={latents.std():.6f}")
                print(f"   Range: [{latents.min():.2f}, {latents.max():.2f}]")
                
                if use_distribution_alignment:
                    # 测试归一化效果
                    with torch.no_grad():
                        normalized = model.normalize_latents(latents)
                        print(f"   归一化后: Mean={normalized.mean():.6f}, Std={normalized.std():.6f}")
                        print(f"   ✅ 分布对齐{'成功' if abs(normalized.std()-1.0)<0.1 else '需要调整'}")
            
            # 获取用户条件
            user_conditions = model.get_user_condition(user_ids)
            
            # 前向传播 - 内部会自动处理归一化
            loss_dict = model.training_step(latents, user_conditions)
            total_loss = loss_dict['total_loss']
            diff_loss = loss_dict['diffusion_loss']
            contrastive_loss = loss_dict['contrastive_loss']
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新统计
            train_losses['total'] += total_loss.item()
            train_losses['diffusion'] += diff_loss.item()
            train_losses['contrastive'] += contrastive_loss.item()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': total_loss.item(),
                'diff': diff_loss.item(),
                'cont': contrastive_loss.item()
            })
        
        # 计算平均损失
        num_batches = len(train_loader)
        for key in train_losses:
            train_losses[key] /= num_batches
        
        # 在epoch结束时更新用户原型
        if (epoch + 1) % 5 == 0:
            print("   🔄 更新用户原型...")
            update_user_prototypes(model, train_loader, device)
        
        # 验证阶段
        model.eval()
        val_losses = {'total': 0.0, 'diffusion': 0.0, 'contrastive': 0.0}
        
        with torch.no_grad():
            for latents, user_ids in val_loader:
                latents = latents.to(device)
                user_ids = user_ids.to(device)
                
                user_conditions = model.get_user_condition(user_ids)
                loss_dict = model.training_step(latents, user_conditions)
                
                val_losses['total'] += loss_dict['total_loss'].item()
                val_losses['diffusion'] += loss_dict['diffusion_loss'].item()
                val_losses['contrastive'] += loss_dict['contrastive_loss'].item()
        
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        # 学习率调度
        scheduler.step()
        
        # 打印epoch总结和分布统计
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_losses['total']:.4f} "
              f"(Diff: {train_losses['diffusion']:.4f}, "
              f"Cont: {train_losses['contrastive']:.4f})")
        print(f"   Val Loss: {val_losses['total']:.4f} "
              f"(Diff: {val_losses['diffusion']:.4f}, "
              f"Cont: {val_losses['contrastive']:.4f})")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 显示当前统计信息
        if use_distribution_alignment and hasattr(model, 'latent_mean'):
            print(f"📈 当前Latent统计 (运行平均):")
            print(f"   Mean: {model.latent_mean:.6f}, Std: {model.latent_std:.6f}")
            print(f"   样本数: {model.n_samples}")
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_path = checkpoint_dir / 'best_model.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'user_prototypes': model.user_prototypes,
                'best_val_loss': best_val_loss,
                'latent_mean': model.latent_mean if hasattr(model, 'latent_mean') else 0.0,
                'latent_std': model.latent_std if hasattr(model, 'latent_std') else 1.0
            }, save_path)
            print(f"   ✅ 保存最佳模型: {save_path} (Val Loss: {best_val_loss:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            save_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'user_prototypes': model.user_prototypes,
                'latent_mean': model.latent_mean if hasattr(model, 'latent_mean') else 0.0,
                'latent_std': model.latent_std if hasattr(model, 'latent_std') else 1.0
            }, save_path)
        
        # 生成样本
        if (epoch + 1) % args.sample_freq == 0:
            generate_samples(model, vae, epoch+1, sample_dir, device, args.num_users)
    
    print("\n🎉 训练完成！")
    
    # 保存最终统计信息
    if use_distribution_alignment:
        stats_path = output_dir / 'latent_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump({
                'latent_mean': float(model.latent_mean),
                'latent_std': float(model.latent_std),
                'n_samples': int(model.n_samples)
            }, f, indent=2)
        print(f"📊 已保存latent统计信息: {stats_path}")
    
    return model


def update_user_prototypes(model, dataloader, device):
    """更新用户原型"""
    user_latents = {}
    
    with torch.no_grad():
        for latents, user_ids in dataloader:
            latents = latents.to(device)
            
            for i, user_id in enumerate(user_ids):
                user_id = user_id.item()
                if user_id not in user_latents:
                    user_latents[user_id] = []
                user_latents[user_id].append(latents[i:i+1])
    
    # 合并每个用户的latents
    for user_id in user_latents:
        user_latents[user_id] = torch.cat(user_latents[user_id], dim=0)
    
    # 更新模型中的原型
    model.update_user_prototypes(user_latents)


def generate_samples(model, vae, epoch, sample_dir, device, num_users):
    """生成并保存样本"""
    print(f"\n🎨 生成样本 (Epoch {epoch})...")
    
    # 选择几个用户生成
    sample_users = list(range(min(4, num_users)))
    
    # 生成latents - 使用分布对齐的生成方法
    with torch.no_grad():
        latents = model.generate(
            user_ids=sample_users,
            num_samples=len(sample_users) * 4,  # 每个用户4个样本
            num_inference_steps=50,
            guidance_scale=7.5  # 使用标准CFG强度
        )
        
        # 显示生成的latent分布
        print(f"📊 生成latent分布: mean={latents.mean():.4f}, std={latents.std():.4f}")
        
        # 如果启用了分布对齐，latent应该已经被反归一化到VAE的分布
        if hasattr(model, 'enable_alignment') and model.enable_alignment:
            print(f"   ✅ 已反归一化到VAE分布 (期望std≈{model.latent_std:.2f})")
        
        # 解码到图像
        print(f"🎨 解码 {len(latents)} 个latent到图像...")
        images = []
        for i in range(0, len(latents), 8):
            batch = latents[i:i+8]
            decoded = vae.decode(batch)
            images.append(decoded)
        
        images = torch.cat(images, dim=0)
    
    # 检查图像范围
    print(f"🔍 图像范围: [{images.min():.3f}, {images.max():.3f}]")
    
    # SimplifiedVAVAE.decode()输出[0,1]范围
    if images.max() > 1.1 or images.min() < -0.1:
        print(f"⚠️ 图像值超出预期范围，进行裁剪")
        images = torch.clamp(images, 0, 1)
    
    # 保存图像网格
    grid = make_grid(images, nrow=4, normalize=False, value_range=(0, 1))
    save_path = sample_dir / f'samples_epoch_{epoch:04d}.png'
    save_image(grid, save_path)
    
    print(f"   ✅ 样本已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='训练分布对齐的扩散模型')
    
    # 数据相关
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/microdoppler',
                      help='原始图像目录')
    parser.add_argument('--latent_dir', type=str, default='/kaggle/working/latents',
                      help='Latent数据目录')
    parser.add_argument('--split_file', type=str, default='/kaggle/working/dataset_split.json',
                      help='数据划分文件')
    parser.add_argument('--prepare_latents', action='store_true', 
                       help='强制重新准备latent数据集')
    
    # VAE相关
    parser.add_argument('--vae_checkpoint', type=str, 
                      default='/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt',
                      help='VA-VAE检查点路径')
    
    # 模型相关
    parser.add_argument('--num_users', type=int, default=31,
                      help='用户数量')
    parser.add_argument('--prototype_dim', type=int, default=768,
                      help='原型维度')
    parser.add_argument('--force_alignment', action='store_true',
                      help='强制启用分布对齐（即使latent已接近标准分布）')
    
    # 训练相关
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--num_users_per_batch', type=int, default=4,
                      help='每批用户数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=2,
                      help='数据加载工作进程数')
    
    # 输出相关
    parser.add_argument('--output_dir', type=str, 
                      default='/kaggle/working/distribution_aligned_diffusion',
                      help='输出目录')
    parser.add_argument('--save_freq', type=int, default=10,
                      help='保存检查点频率')
    parser.add_argument('--sample_freq', type=int, default=5,
                      help='生成样本频率')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 训练模型
    model = train_distribution_aligned_diffusion(args)
    
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()
