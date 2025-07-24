#!/usr/bin/env python3
"""
阶段2: 用户条件化DiT训练
在潜在空间训练扩散模型，遵循LightningDiT的train.py实现
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
import math

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from models.lightningdit import LightningDiT
from transport import create_transport

class LatentDataset(Dataset):
    """
    潜在特征数据集
    遵循LightningDiT的ImgLatentDataset实现
    """
    
    def __init__(self, latent_file, latent_norm=True, latent_multiplier=1.0):
        print(f"📊 加载潜在特征: {latent_file}")
        
        # 使用safetensors加载数据
        with safe_open(latent_file, framework="pt", device="cpu") as f:
            self.latents = f.get_tensor('latents')  # (N, 32, 16, 16)
            self.user_ids = f.get_tensor('user_ids')  # (N,)
            
            # 读取元数据
            self.num_samples = f.get_tensor('num_samples').item()
            self.num_users = f.get_tensor('num_users').item()
        
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        print(f"  样本数量: {len(self.latents)}")
        print(f"  特征形状: {self.latents.shape}")
        print(f"  用户数量: {self.num_users}")
        print(f"  用户ID范围: [{self.user_ids.min()}, {self.user_ids.max()}]")
        
        # 计算潜在特征的统计信息（用于归一化）
        if self.latent_norm:
            self._compute_latent_stats()
    
    def _compute_latent_stats(self):
        """加载或计算潜在特征的均值和标准差"""
        print("📈 加载潜在特征统计信息...")

        # 首先尝试加载预计算的统计信息
        stats_file = Path(self.latent_file).parent / "latents_stats.pt"

        if stats_file.exists():
            print(f"📊 加载统计信息: {stats_file}")
            stats = torch.load(stats_file)
            self.latent_mean = stats['mean']  # (1, 32, 1, 1)
            self.latent_std = stats['std']    # (1, 32, 1, 1)

            print(f"  使用预计算的统计信息")
            print(f"  均值形状: {self.latent_mean.shape}")
            print(f"  标准差形状: {self.latent_std.shape}")
        else:
            print("⚠️  未找到预计算的统计信息，使用全局统计")
            # 回退到全局统计信息
            self.latent_mean = self.latents.mean()
            self.latent_std = self.latents.std()

            print(f"  全局均值: {self.latent_mean:.4f}")
            print(f"  全局标准差: {self.latent_std:.4f}")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx].clone()  # (32, 16, 16)
        user_id = self.user_ids[idx].item()
        
        # 应用归一化
        if self.latent_norm:
            latent = (latent - self.latent_mean) / self.latent_std
        
        # 应用缩放因子
        latent = latent * self.latent_multiplier
        
        return {
            'latent': latent,
            'user_id': user_id,
            'y': user_id - 1  # 转换为0-based索引，用于DiT的类别条件
        }

class UserConditionedDiT(pl.LightningModule):
    """
    用户条件化的DiT模型
    基于LightningDiT实现，添加用户条件
    """
    
    def __init__(
        self,
        num_users,
        input_size=16,  # 16x16 latent
        patch_size=2,
        in_channels=32,  # VA-VAE的潜在维度
        hidden_size=1152,  # DiT-XL配置
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        lr=1e-4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_users = num_users
        self.lr = lr
        
        # 创建DiT模型
        self.dit = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=0.1,  # 用于classifier-free guidance
            num_classes=num_users,   # 用户作为类别条件
            learn_sigma=learn_sigma,
        )
        
        # 创建扩散传输
        self.transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            loss_weight=None,
            train_eps=1e-5,
            sample_eps=1e-4,
        )
        
        print(f"🤖 创建用户条件化DiT模型:")
        print(f"  用户数量: {num_users}")
        print(f"  输入尺寸: {input_size}x{input_size}")
        print(f"  潜在维度: {in_channels}")
        print(f"  隐藏维度: {hidden_size}")
        print(f"  层数: {depth}")
        print(f"  注意力头数: {num_heads}")
    
    def forward(self, x, t, y):
        """
        前向传播
        Args:
            x: (B, 32, 16, 16) 噪声潜在特征
            t: (B,) 时间步
            y: (B,) 用户ID (0-based)
        """
        return self.dit(x, t, y)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        latents = batch['latent']  # (B, 32, 16, 16)
        user_ids = batch['y']      # (B,) 0-based用户ID
        
        # 扩散训练
        model_kwargs = dict(y=user_ids)
        loss_dict = self.transport.training_losses(self.dit, latents, model_kwargs)
        loss = loss_dict["loss"].mean()
        
        # 记录损失
        self.log('train/loss', loss, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        latents = batch['latent']
        user_ids = batch['y']
        
        model_kwargs = dict(y=user_ids)
        loss_dict = self.transport.training_losses(self.dit, latents, model_kwargs)
        loss = loss_dict["loss"].mean()
        
        self.log('val/loss', loss, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

class MicroDopplerDataModule(pl.LightningDataModule):
    """
    微多普勒数据模块
    """
    
    def __init__(
        self,
        train_latent_file,
        val_latent_file,
        batch_size=32,
        num_workers=4,
        latent_norm=True,
        latent_multiplier=1.0
    ):
        super().__init__()
        self.train_latent_file = train_latent_file
        self.val_latent_file = val_latent_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
    
    def setup(self, stage=None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = LatentDataset(
                self.train_latent_file,
                latent_norm=self.latent_norm,
                latent_multiplier=self.latent_multiplier
            )
            self.val_dataset = LatentDataset(
                self.val_latent_file,
                latent_norm=self.latent_norm,
                latent_multiplier=self.latent_multiplier
            )
            
            # 确保用户数量一致
            assert self.train_dataset.num_users == self.val_dataset.num_users
            self.num_users = self.train_dataset.num_users
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

def main():
    parser = argparse.ArgumentParser(description='训练用户条件化DiT模型')
    parser.add_argument('--latent_dir', type=str, required=True,
                       help='潜在特征目录 (包含train.safetensors和val.safetensors)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--devices', type=int, default=1,
                       help='GPU数量')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       help='训练精度')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    print("🎯 用户条件化DiT训练 - 阶段2")
    print("=" * 50)
    
    # 创建数据模块
    data_module = MicroDopplerDataModule(
        train_latent_file=os.path.join(args.latent_dir, 'train.safetensors'),
        val_latent_file=os.path.join(args.latent_dir, 'val.safetensors'),
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # 设置数据模块以获取用户数量
    data_module.setup()
    
    # 创建模型
    model = UserConditionedDiT(
        num_users=data_module.num_users,
        lr=args.lr
    )
    
    # 设置回调
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='dit-{epoch:02d}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator='gpu',
        strategy='ddp' if args.devices > 1 else 'auto',
        precision=args.precision,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=args.output_dir
    )
    
    print("🚀 开始训练...")
    trainer.fit(model, data_module)
    
    print("✅ 训练完成!")

if __name__ == "__main__":
    main()
