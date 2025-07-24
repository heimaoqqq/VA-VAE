#!/usr/bin/env python3
"""
基于Accelerate的DiT训练脚本
参考LightningDiT原项目的多GPU训练方式
"""

import os
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from tqdm import tqdm
import math

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from safetensors import safe_open

# 导入LightningDiT组件
from LightningDiT.models.lightningdit import LightningDiT_models
from LightningDiT.transport import create_transport

class LatentDataset(torch.utils.data.Dataset):
    """潜在特征数据集"""
    
    def __init__(self, latent_file, latent_norm=True, latent_multiplier=1.0):
        print(f"📊 加载潜在特征: {latent_file}")
        
        # 保存文件路径
        self.latent_file = latent_file
        
        # 使用safetensors加载数据
        with safe_open(latent_file, framework="pt", device="cpu") as f:
            self.latents = f.get_tensor('latents')  # (N, 32, 16, 16)
            self.user_ids = f.get_tensor('user_ids')  # (N,)
            
            # 读取元数据
            self.num_samples = f.get_tensor('num_samples').item()
            self.num_users = f.get_tensor('num_users').item()
        
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        print(f"  样本数量: {self.num_samples}")
        print(f"  特征形状: {self.latents.shape}")
        print(f"  用户数量: {self.num_users}")
        print(f"  用户ID范围: [{self.user_ids.min().item()}, {self.user_ids.max().item()}]")
        
        # 计算统计信息
        self._compute_latent_stats()
    
    def _compute_latent_stats(self):
        """加载或计算潜在特征的均值和标准差"""
        print("📈 加载潜在特征统计信息...")
        
        latent_dir = Path(self.latent_file).parent
        
        # 检查推荐使用哪个统计信息
        recommendation_file = latent_dir / "stats_recommendation.txt"
        use_imagenet = False
        
        if recommendation_file.exists():
            with open(recommendation_file, 'r') as f:
                content = f.read()
                if "imagenet" in content.lower():
                    use_imagenet = True
                    print("📋 推荐使用ImageNet统计信息")
        
        # 选择统计信息文件
        if use_imagenet:
            stats_file = latent_dir / "latents_stats_imagenet.pt"
            stats_type = "ImageNet"
        else:
            stats_file = latent_dir / "latents_stats.pt"
            stats_type = "微多普勒"
        
        if stats_file.exists():
            print(f"📊 加载{stats_type}统计信息: {stats_file}")
            stats = torch.load(stats_file)
            self.latent_mean = stats['mean']  # (1, 32, 1, 1)
            self.latent_std = stats['std']    # (1, 32, 1, 1)
            
            print(f"  使用{stats_type}统计信息")
            print(f"  均值形状: {self.latent_mean.shape}")
            print(f"  标准差形状: {self.latent_std.shape}")
            print(f"  全局均值: {self.latent_mean.mean():.4f}")
            print(f"  全局标准差: {self.latent_std.mean():.4f}")
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
            # 确保统计信息的形状匹配
            # latent: (32, 16, 16), mean/std: (1, 32, 1, 1)
            # 需要squeeze掉第一个维度
            mean = self.latent_mean.squeeze(0)  # (32, 1, 1)
            std = self.latent_std.squeeze(0)    # (32, 1, 1)
            latent = (latent - mean) / std
        
        # 应用缩放因子
        latent = latent * self.latent_multiplier
        
        # 确保返回的latent是3维的 (C, H, W)
        if len(latent.shape) != 3:
            print(f"⚠️  警告: latent维度异常 {latent.shape}, 尝试修复")
            latent = latent.squeeze()  # 移除所有大小为1的维度
            if len(latent.shape) != 3:
                raise ValueError(f"无法修复latent维度: {latent.shape}")
        
        return {
            'latent': latent,
            'user_id': user_id,
            'y': user_id - 1  # 转换为0-based索引，用于DiT的类别条件
        }

def main():
    parser = argparse.ArgumentParser(description='基于Accelerate的DiT训练')
    parser.add_argument('--latent_dir', type=str, required=True, help='潜在特征目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 初始化Accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',  # 使用混合精度
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    print("🎯 基于Accelerate的用户条件化DiT训练")
    print("=" * 60)
    print(f"🔧 Accelerator配置:")
    print(f"  进程数: {accelerator.num_processes}")
    print(f"  当前进程: {accelerator.process_index}")
    print(f"  设备: {accelerator.device}")
    print(f"  混合精度: {accelerator.mixed_precision}")
    print(f"  分布式: {accelerator.distributed_type}")
    
    # 创建数据集
    train_dataset = LatentDataset(
        latent_file=os.path.join(args.latent_dir, 'train.safetensors'),
        latent_norm=True,
        latent_multiplier=1.0
    )
    
    val_dataset = LatentDataset(
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
    
    # 创建模型
    model = LightningDiT_models['LightningDiT-XL/1'](
        input_size=16,  # 16x16 latent
        num_classes=train_dataset.num_users,  # 用户数量作为类别数
        in_channels=32,  # 32通道
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False
    )
    
    # 创建transport (扩散过程)
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0
    )
    
    # 使用Accelerator准备模型、优化器和数据加载器
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    print("🚀 开始训练...")
    
    # 训练循环
    for epoch in range(args.max_epochs):
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
                
                # 扩散训练
                model_kwargs = dict(y=user_ids)
                loss_dict = transport.training_losses(model, latents, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(step+1):.4f}'
                })
        
        # 验证
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
            print(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_dataloader):.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}"))
    
    print("✅ 训练完成!")

if __name__ == "__main__":
    main()
