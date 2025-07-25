#!/usr/bin/env python3
"""
阶段2: DiT训练
基于LightningDiT原项目的train.py
使用Accelerate进行分布式训练
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import math
from datetime import datetime

from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open

# 导入LightningDiT组件
import sys
import os

# 确保正确的路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
lightningdit_path = os.path.join(current_dir, 'LightningDiT')
if lightningdit_path not in sys.path:
    sys.path.append(lightningdit_path)

from models.lightningdit import LightningDiT_models
from transport import create_transport
from datasets.img_latent_dataset import ImgLatentDataset

class MicroDopplerLatentDataset(torch.utils.data.Dataset):
    """微多普勒潜在特征数据集 (基于原项目ImgLatentDataset)"""
    
    def __init__(self, latent_file, latent_norm=True, latent_multiplier=1.0):
        print(f"📊 加载潜在特征: {latent_file}")
        
        # 使用safetensors加载数据
        with safe_open(latent_file, framework="pt", device="cpu") as f:
            self.latents = f.get_tensor('latents')  # (N, 32, 16, 16)
            self.user_ids = f.get_tensor('user_ids')  # (N,)
            self.num_samples = f.get_tensor('num_samples').item()
            self.num_users = f.get_tensor('num_users').item()
        
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        print(f"  样本数量: {self.num_samples}")
        print(f"  特征形状: {self.latents.shape}")
        print(f"  用户数量: {self.num_users}")
        
        # 加载统计信息 (参考原项目)
        self._load_latent_stats(Path(latent_file).parent)
    
    def _load_latent_stats(self, data_dir):
        """加载潜在特征统计信息"""
        stats_file = data_dir / "latents_stats.pt"
        if stats_file.exists():
            print(f"📊 加载统计信息: {stats_file}")
            stats = torch.load(stats_file)
            self.latent_mean = stats['mean']  # (1, 32, 1, 1)
            self.latent_std = stats['std']    # (1, 32, 1, 1)
            print(f"  均值形状: {self.latent_mean.shape}")
            print(f"  标准差形状: {self.latent_std.shape}")
        else:
            print("⚠️  未找到统计信息，使用全局统计")
            self.latent_mean = self.latents.mean()
            self.latent_std = self.latents.std()
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx].clone()  # (32, 16, 16)
        user_id = self.user_ids[idx].item()
        
        # 应用归一化 (参考原项目)
        if self.latent_norm:
            mean = self.latent_mean.squeeze(0)  # (32, 1, 1)
            std = self.latent_std.squeeze(0)    # (32, 1, 1)
            latent = (latent - mean) / std
        
        # 应用缩放因子
        latent = latent * self.latent_multiplier
        
        return {
            'latent': latent,
            'y': user_id - 1  # 转换为0-based索引
        }

def main():
    parser = argparse.ArgumentParser(description='基于Accelerate的DiT训练')
    parser.add_argument('--latent_dir', type=str, required=True, help='潜在特征目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--model_name', type=str, default='LightningDiT-XL/1', help='模型名称')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_every', type=int, default=10, help='保存间隔')
    
    args = parser.parse_args()
    
    # 初始化Accelerator (参考原项目配置)
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    if accelerator.is_main_process:
        print("🎯 基于Accelerate的用户条件化DiT训练")
        print("=" * 60)
        print(f"🔧 Accelerator配置:")
        print(f"  进程数: {accelerator.num_processes}")
        print(f"  当前进程: {accelerator.process_index}")
        print(f"  设备: {accelerator.device}")
        print(f"  混合精度: {accelerator.mixed_precision}")
        print(f"  分布式类型: {accelerator.distributed_type}")
    
    # 创建数据集 (参考原项目)
    train_dataset = MicroDopplerLatentDataset(
        latent_file=os.path.join(args.latent_dir, 'train.safetensors'),
        latent_norm=True,
        latent_multiplier=1.0
    )
    
    val_dataset = MicroDopplerLatentDataset(
        latent_file=os.path.join(args.latent_dir, 'val.safetensors'),
        latent_norm=True,
        latent_multiplier=1.0
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型 (参考原项目)
    model = LightningDiT_models[args.model_name](
        input_size=16,  # 16x16 latent
        num_classes=train_dataset.num_users,  # 用户数量作为类别数
        in_channels=32,  # 32通道
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False
    )
    
    if accelerator.is_main_process:
        print(f"📊 模型信息:")
        print(f"  模型: {args.model_name}")
        print(f"  输入尺寸: 16x16")
        print(f"  输入通道: 32")
        print(f"  类别数: {train_dataset.num_users}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 创建transport (扩散过程，参考原项目)
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None
    )
    
    # 创建优化器 (参考原项目)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0
    )
    
    # 学习率调度器 (参考原项目)
    num_training_steps = len(train_dataloader) * args.max_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps
    )
    
    # 使用Accelerator准备模型、优化器和数据加载器
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    if accelerator.is_main_process:
        print("🚀 开始训练...")
        print(f"  训练样本: {len(train_dataset)}")
        print(f"  验证样本: {len(val_dataset)}")
        print(f"  批次大小: {args.batch_size}")
        print(f"  总批次数: {len(train_dataloader)}")
        print(f"  训练轮数: {args.max_epochs}")
    
    # 训练循环 (参考原项目)
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.max_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{args.max_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                latents = batch['latent']  # (B, 32, 16, 16)
                user_ids = batch['y']      # (B,) 0-based用户ID
                
                # 扩散训练 (参考原项目)
                model_kwargs = dict(y=user_ids)
                loss_dict = transport.training_losses(model, latents, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪 (参考原项目)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(step+1):.4f}',
                    'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
                })
        
        # 验证阶段
        if accelerator.is_main_process:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    latents = batch['latent']
                    user_ids = batch['y']
                    
                    model_kwargs = dict(y=user_ids)
                    loss_dict = transport.training_losses(model, latents, model_kwargs)
                    val_loss += loss_dict["loss"].mean().item()
            
            val_loss /= len(val_dataloader)
            train_loss = total_loss / len(train_dataloader)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                accelerator.save_state(os.path.join(args.output_dir, "best_model"))
                print(f"✅ 保存最佳模型 (Val Loss: {val_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % args.save_every == 0:
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}"))
                print(f"💾 保存检查点: epoch_{epoch+1}")
    
    if accelerator.is_main_process:
        print("✅ 训练完成!")
        print(f"最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
