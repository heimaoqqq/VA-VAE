#!/usr/bin/env python3
"""
训练增强条件扩散模型的完整脚本
整合SimplifiedVAVAE + Enhanced Conditional Diffusion + 平衡数据加载
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
from enhanced_conditional_diffusion import EnhancedConditionalDiffusion
from simplified_vavae import SimplifiedVAVAE

def compute_latent_statistics(dataloader, device, max_batches=50):
    """
    计算训练数据的latent分布统计
    Args:
        dataloader: 数据加载器
        device: 设备
        max_batches: 最大批次数（避免计算过久）
    Returns:
        dict: {'mean': float, 'std': float, 'min': float, 'max': float}
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

def train_enhanced_diffusion(args):
    """训练增强条件扩散模型"""
    
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
    
    # 初始化VAE
    vae_path = args.vae_checkpoint if hasattr(args, 'vae_checkpoint') else "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    vae = SimplifiedVAVAE(vae_path)
    vae.eval()
    vae.freeze()
    vae = vae.to(device)
    print(f"✅ VAE加载完成，缩放因子: {vae.scale_factor}")
    vae.eval()
    
    # 步骤2: 智能检查并准备latent数据集
    latent_dir = Path(args.latent_dir)
    
    # 检查latent数据集是否已经存在且完整
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
        print("   跳过数据集准备步骤")
    
    # 步骤3: 创建数据加载器
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
    
    # 步骤3.5: 计算训练数据的实际latent分布统计
    print("\n📊 计算训练数据latent分布统计...")
    latent_stats = compute_latent_statistics(train_loader, device, max_batches=50)
    train_mean = latent_stats['mean']
    train_std = latent_stats['std']
    latent_min = latent_stats['min']
    latent_max = latent_stats['max']
    
    print(f"📈 实际训练latent分布:")
    print(f"   Mean: {train_mean:.6f}")
    print(f"   Std:  {train_std:.6f}")
    print(f"   Range: [{latent_min:.2f}, {latent_max:.2f}]")
    print(f"   3σ Range: [{train_mean-3*train_std:.2f}, {train_mean+3*train_std:.2f}]")
    
    # 步骤4: 创建增强扩散模型
    print("\n🚀 创建增强条件扩散模型...")
    model = EnhancedConditionalDiffusion(
        vae=vae,
        num_users=args.num_users,
        prototype_dim=args.prototype_dim,
        latent_mean=latent_stats['mean'],
        latent_std=latent_stats['std']
    )
    
    # 关键修复：传递VAE实例以获取正确的scale_factor
    model.vae = vae
    print(f"✅ 已将VAE实例传递给扩散模型 (scale_factor={vae.scale_factor})")
    
    # 🔧 使用动态计算的训练分布
    model.set_training_stats(latent_stats['mean'], latent_stats['std'])
    
    print(f"📊 VAE配置: scale_factor={vae.scale_factor}")
    print(f"🎯 扩散模型: num_users={args.num_users}, prototype_dim={args.prototype_dim}")
    print(f"🔑 标准化方法: 缩放因子={model.scale_factor:.4f} (类似Stable Diffusion)")
    
    # 步骤5: 设置优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs
    )
    
    # 步骤6: 训练循环
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
            
            # 第一个epoch的第一个batch显示latent分布信息
            if epoch == 0 and batch_idx == 0:
                print(f"\n📊 训练数据Latent分布统计:")
                print(f"   Shape: {latents.shape}")
                print(f"   Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")
                print(f"   Range: [{latents.min():.2f}, {latents.max():.2f}]")
            
            # 前向传播
            losses = model.training_step(
                latents, user_ids, 
                support_ratio=args.support_ratio
            )
            
            # 反向传播
            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新统计
            for key in train_losses:
                loss_key = f'{key}_loss' if key != 'total' else 'total_loss'
                train_losses[key] += losses[loss_key].item()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': losses['total_loss'].item(),
                'diff': losses['diffusion_loss'].item(),
                'cont': losses['contrastive_loss'].item()
            })
            
            # 使用均值作为用户原型 - 修复噪声初始化更新（在epoch结束时统一更新更高效）
            # if batch_idx % args.prototype_update_freq == 0:
            #     update_user_prototypes(model, train_loader, device)
        
        # 计算平均损失
        num_batches = len(train_loader)
        for key in train_losses:
            train_losses[key] /= num_batches
        
        # 在epoch结束时更新用户原型（更高效）
        if (epoch + 1) % 5 == 0:  # 每5个epoch更新一次
            print("   🔄 更新用户原型...")
            update_user_prototypes(model, train_loader, device)
        
        # 验证阶段
        model.eval()
        val_losses = {'total': 0.0, 'diffusion': 0.0, 'contrastive': 0.0}
        
        with torch.no_grad():
            for latents, user_ids in val_loader:
                latents = latents.to(device)
                user_ids = user_ids.to(device)
                
                losses = model.training_step(latents, user_ids)
                
                for key in val_losses:
                    loss_key = f'{key}_loss' if key != 'total' else 'total_loss'
                    loss_value = losses[loss_key]
                    # 处理可能已经是float的情况
                    if hasattr(loss_value, 'item'):
                        val_losses[key] += loss_value.item()
                    else:
                        val_losses[key] += float(loss_value)
        
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        # 学习率调度
        scheduler.step()
        
        # 打印epoch总结
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_losses['total']:.4f} "
              f"(Diff: {train_losses['diffusion']:.4f}, "
              f"Cont: {train_losses['contrastive']:.4f})")
        print(f"   Val Loss: {val_losses['total']:.4f} "
              f"(Diff: {val_losses['diffusion']:.4f}, "
              f"Cont: {val_losses['contrastive']:.4f})")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_path = checkpoint_dir / 'best_model.pth'
            
            # 删除旧的最佳模型
            if save_path.exists():
                save_path.unlink()
                print(f"   🗑️ 删除旧的最佳模型")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'user_prototypes': model.user_prototypes,
                'best_val_loss': best_val_loss
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
                'user_prototypes': model.user_prototypes
            }, save_path)
        
        # 每个epoch都生成4x4样本网格
        generate_samples(model, vae, epoch+1, sample_dir, device, args.num_users)
    
    print("\n🎉 训练完成！")
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
    print(f"\n🎨 生成样本 (Epoch {epoch+1})...")
    
    # 选择几个用户生成
    sample_users = list(range(min(4, num_users)))
    
    # 生成latents
    with torch.no_grad():
        latents = model.generate(
            user_ids=sample_users,
            num_samples_per_user=4,
            num_inference_steps=100,  # 使用100步确保生成质量
            guidance_scale=4.0,       # 使用标准CFG强度
            use_ddim=True
        )
        
        # 关键分布验证信息
        print(f"📊 生成latent分布: mean={latents.mean():.4f}, std={latents.std():.4f}")
        print(f"   ✅ 修复验证: 期望std≈1.54 {'✅' if abs(latents.std() - 1.54) < 0.3 else '❌'}")
        
        # 解码到图像
        print(f"🎨 解码 {len(latents)} 个latent到图像...")
        images = []
        for i in range(0, len(latents), 8):
            batch = latents[i:i+8]
            decoded = vae.decode(batch)
            images.append(decoded)
        
        images = torch.cat(images, dim=0)
    
    # 检查图像值范围
    print(f"🔍 解码图像范围: min={images.min():.3f}, max={images.max():.3f}")
    print(f"🔍 图像形状: {images.shape}")
    print(f"🔍 latent范围: min={latents.min():.3f}, max={latents.max():.3f}, std={latents.std():.3f}")
    
    # SimplifiedVAVAE.decode()已经输出[0,1]范围，无需再处理
    # 只需确保在合理范围内
    if images.max() > 1.1 or images.min() < -0.1:
        print(f"⚠️ 图像值超出预期范围，进行裁剪")
        images = torch.clamp(images, 0, 1)
    else:
        print("✅ 图像已在[0,1]范围，无需处理")
    
    # 保存图像网格
    grid = make_grid(images, nrow=4, normalize=False, value_range=(0, 1))
    save_path = sample_dir / f'samples_epoch_{epoch:04d}.png'
    save_image(grid, save_path)
    
    print(f"   ✅ 样本已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='训练增强条件扩散模型')
    
    # 数据相关
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/microdoppler',
                      help='原始图像目录')
    parser.add_argument('--latent_dir', type=str, default='/kaggle/working/latents',
                      help='Latent数据目录')
    parser.add_argument('--split_file', type=str, default='/kaggle/working/dataset_split.json',
                      help='数据划分文件')
    parser.add_argument('--prepare_latents', action='store_true', 
                       help='准备latent数据集（如果需要）')
    # UNet方法：使用简化的匹配训练分布方案
    
    parser.add_argument('--vae_checkpoint', type=str, 
                      default='/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt',
                      help='VA-VAE检查点路径')
    parser.add_argument('--num_users', type=int, default=31,
                      help='用户数量')
    parser.add_argument('--prototype_dim', type=int, default=768,
                      help='原型维度')
    
    # 训练相关
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--num_users_per_batch', type=int, default=4,
                      help='每批用户数')
    parser.add_argument('--support_ratio', type=float, default=0.3,
                      help='Support set比例')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=2,
                      help='数据加载工作进程数')
    
    # 其他
    parser.add_argument('--output_dir', type=str, 
                      default='/kaggle/working/enhanced_diffusion',
                      help='输出目录')
    parser.add_argument('--save_freq', type=int, default=10,
                      help='保存检查点频率')
    parser.add_argument('--sample_freq', type=int, default=5,
                      help='生成样本频率')
    parser.add_argument('--prototype_update_freq', type=int, default=50,
                      help='原型更新频率（批次）')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 训练模型
    model = train_enhanced_diffusion(args)
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
