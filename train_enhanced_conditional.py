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
from torchvision.utils import save_image, make_grid

from simplified_vavae import SimplifiedVAVAE
from enhanced_conditional_diffusion import EnhancedConditionalDiffusion
from microdoppler_data_loader import (
    create_balanced_dataloader, 
    prepare_latent_dataset
)


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
    
    # 步骤1: 准备VAE和latent数据
    print("\n📦 加载VA-VAE模型...")
    vae = SimplifiedVAVAE(args.vae_checkpoint)
    vae = vae.to(device)
    vae.eval()
    
    # 步骤2: 准备latent数据集（如果需要）
    latent_dir = Path(args.latent_dir)
    if not latent_dir.exists() or args.prepare_latents:
        print("\n🔄 准备latent数据集...")
        latent_dir = prepare_latent_dataset(
            image_dir=args.image_dir,
            vae_model=vae,
            output_dir=args.latent_dir,
            split_file=args.split_file,
            device=device
        )
    
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
    
    # 步骤4: 创建增强扩散模型
    print("\n🚀 创建增强条件扩散模型...")
    model = EnhancedConditionalDiffusion(
        num_users=args.num_users,
        prototype_dim=args.prototype_dim
    )
    model = model.to(device)
    
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
            
            # 定期更新用户原型
            if batch_idx % args.prototype_update_freq == 0:
                update_user_prototypes(model, train_loader, device)
        
        # 计算平均损失
        num_batches = len(train_loader)
        for key in train_losses:
            train_losses[key] /= num_batches
        
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
                    val_losses[key] += losses[loss_key].item()
        
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'user_prototypes': model.user_prototypes
            }, save_path)
            print(f"   ✅ 保存最佳模型: {save_path}")
        
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
        
        # 生成样本
        if (epoch + 1) % args.sample_freq == 0:
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
    print(f"\n🎨 生成样本 (Epoch {epoch})...")
    
    # 选择几个用户生成
    sample_users = list(range(min(4, num_users)))
    
    # 生成latents
    with torch.no_grad():
        latents = model.generate(
            user_ids=sample_users,
            num_samples_per_user=4,
            num_inference_steps=100,
            guidance_scale=2.0,
            use_ddim=True
        )
        
        # 解码到图像
        images = []
        for i in range(0, len(latents), 8):
            batch = latents[i:i+8]
            decoded = vae.decode(batch)
            images.append(decoded)
        
        images = torch.cat(images, dim=0)
    
    # 保存图像网格
    grid = make_grid(images, nrow=4, normalize=True, value_range=(0, 1))
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
    parser.add_argument('--split_file', type=str, default='/kaggle/working/data_split.json',
                      help='数据划分文件')
    parser.add_argument('--prepare_latents', action='store_true',
                      help='是否准备latent数据集')
    
    # 模型相关
    parser.add_argument('--vae_checkpoint', type=str, 
                      default='/kaggle/working/checkpoints/va_vae_final.ckpt',
                      help='VA-VAE检查点路径')
    parser.add_argument('--num_users', type=int, default=31,
                      help='用户数量')
    parser.add_argument('--prototype_dim', type=int, default=256,
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
