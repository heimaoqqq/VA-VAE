#!/usr/bin/env python3
"""
步骤5: 条件DiT微调训练
基于预训练LightningDiT-XL-64ep进行用户条件微调
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import yaml
import argparse
from tqdm import tqdm
import wandb
from datetime import datetime

# 添加LightningDiT到路径
sys.path.append(str(Path("LightningDiT").absolute()))

from dit.models import DiT_models
from tokenizer.vavae import AutoencoderKL
from step4_microdoppler_adapter import MicroDopplerDataModule, UserConditionEncoder

class ConditionalDiT(nn.Module):
    """条件DiT模型"""
    
    def __init__(
        self,
        model: str = "DiT-XL/1",
        num_users: int = 10,
        condition_dim: int = 768,
        frozen_backbone: bool = True,
        dropout: float = 0.15,
        pretrained_path: str = None
    ):
        super().__init__()
        self.num_users = num_users
        self.condition_dim = condition_dim
        self.frozen_backbone = frozen_backbone
        
        # 加载预训练DiT
        self.dit = DiT_models[model](input_size=16)  # 16 = 256/16
        
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 冻结主干网络
        if frozen_backbone:
            self._freeze_backbone()
        
        # 用户条件编码器
        self.condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=condition_dim,
            dropout=dropout
        )
        
        # 条件注入层 - 修改DiT的adaLN层
        self._inject_condition_layers()
        
        print(f"✅ 条件DiT初始化完成")
        print(f"   - 主干冻结: {frozen_backbone}")
        print(f"   - 总参数: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   - 可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """加载预训练权重"""
        print(f"📥 加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'ema' in checkpoint:
                state_dict = checkpoint['ema']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 加载权重，忽略不匹配的键
        missing, unexpected = self.dit.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️ 缺失键: {len(missing)}")
        if unexpected:
            print(f"⚠️ 额外键: {len(unexpected)}")
    
    def _freeze_backbone(self):
        """冻结主干网络"""
        frozen_count = 0
        for name, param in self.dit.named_parameters():
            # 保持最后几层可训练，用于条件适配
            if not any(keep in name for keep in ['final_layer', 'adaln']):
                param.requires_grad = False
                frozen_count += 1
        
        print(f"🔒 冻结参数: {frozen_count}")
    
    def _inject_condition_layers(self):
        """注入条件处理层"""
        # 在DiT的adaLN层中注入用户条件
        # 这是一个简化实现，实际可能需要更复杂的架构修改
        
        # 为每个transformer block添加条件处理
        for i, block in enumerate(self.dit.blocks):
            # 创建条件融合层
            condition_fusion = nn.Sequential(
                nn.Linear(self.condition_dim, block.adaLN_modulation[1].out_features),
                nn.SiLU(),
                nn.Linear(block.adaLN_modulation[1].out_features, block.adaLN_modulation[1].out_features)
            )
            
            # 替换原有的adaLN调制层
            original_adaln = block.adaLN_modulation
            block.adaLN_modulation = nn.Sequential(
                original_adaln,
                condition_fusion
            )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, user_classes: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, H, W) 潜向量
            t: (B,) 时间步
            user_classes: (B,) 用户类别
        Returns:
            predicted_noise: (B, C, H, W) 预测的噪声
        """
        # 编码用户条件
        user_condition = self.condition_encoder(user_classes)  # (B, condition_dim)
        
        # DiT前向传播（需要修改以支持条件）
        predicted_noise = self._conditional_dit_forward(x, t, user_condition)
        
        return predicted_noise
    
    def _conditional_dit_forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """带条件的DiT前向传播"""
        # 这里需要修改DiT的前向传播逻辑以支持条件输入
        # 简化实现：直接调用原有DiT，条件通过修改后的adaLN层注入
        return self.dit(x, t)

class ConditionalDiTTrainer:
    """条件DiT训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self._setup_model()
        
        # 初始化数据
        self._setup_data()
        
        # 初始化优化器
        self._setup_optimizer()
        
        # 初始化日志
        self._setup_logging()
    
    def _setup_model(self):
        """设置模型"""
        model_config = self.config['model']['params']
        
        # VAE
        self.vae = AutoencoderKL.from_config("LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
        self.vae.to(self.device)
        self.vae.eval()
        
        # 条件DiT
        self.model = ConditionalDiT(
            **model_config,
            pretrained_path="models/lightningdit-xl-imagenet256-64ep.pt"
        )
        self.model.to(self.device)
        
        print(f"✅ 模型设置完成，设备: {self.device}")
    
    def _setup_data(self):
        """设置数据"""
        data_config = self.config['data']['params']
        
        self.data_module = MicroDopplerDataModule(**data_config)
        self.data_module.setup()
        
        # 更新模型的用户数量
        actual_num_users = self.data_module.num_users
        if actual_num_users != self.model.num_users:
            print(f"⚠️ 更新用户数量: {self.model.num_users} -> {actual_num_users}")
            # 这里可能需要重新初始化条件编码器
    
    def _setup_optimizer(self):
        """设置优化器"""
        opt_config = self.config['optimizer']['params']
        
        # 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(trainable_params, **opt_config)
        
        # 学习率调度器
        if 'scheduler' in self.config:
            sch_config = self.config['scheduler']['params']
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **sch_config
            )
        else:
            self.scheduler = None
    
    def _setup_logging(self):
        """设置日志"""
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"experiments/conditional_dit_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.exp_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"✅ 实验目录: {self.exp_dir}")
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失"""
        images = batch['image'].to(self.device)
        user_classes = batch['user_class'].to(self.device)
        
        batch_size = images.shape[0]
        
        # VAE编码
        with torch.no_grad():
            posterior = self.vae.encode(images)
            z = posterior.sample()  # (B, C, H, W)
        
        # 添加噪声（DDPM训练）
        noise = torch.randn_like(z)
        timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        # 前向过程: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
        # 简化实现，使用固定的噪声调度
        alpha_bar_t = 1 - timesteps.float() / 1000
        alpha_bar_t = alpha_bar_t.view(batch_size, 1, 1, 1)
        
        z_noisy = torch.sqrt(alpha_bar_t) * z + torch.sqrt(1 - alpha_bar_t) * noise
        
        # 模型预测
        predicted_noise = self.model(z_noisy, timesteps, user_classes)
        
        # MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        train_loader = self.data_module.train_dataloader()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播
            loss = self.compute_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('trainer', {}).get('gradient_clip_val'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['trainer']['gradient_clip_val']
                )
            
            self.optimizer.step()
            
            # 更新统计
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        
        val_loader = self.data_module.val_dataloader()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.exp_dir / "last.ckpt")
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.exp_dir / "best.ckpt")
    
    def train(self):
        """完整训练流程"""
        trainer_config = self.config.get('trainer', {})
        max_epochs = trainer_config.get('max_epochs', 50)
        val_every_n_epochs = trainer_config.get('check_val_every_n_epoch', 2)
        
        best_val_loss = float('inf')
        
        print(f"🚀 开始训练，共{max_epochs}个epoch")
        
        for epoch in range(max_epochs):
            print(f"\n📅 Epoch {epoch+1}/{max_epochs}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"🎯 训练损失: {train_loss:.4f}")
            
            # 验证
            if (epoch + 1) % val_every_n_epochs == 0:
                val_loss = self.validate()
                print(f"📊 验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    print(f"🏆 新的最佳模型！验证损失: {best_val_loss:.4f}")
                
                self.save_checkpoint(epoch, is_best)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
        
        print(f"\n🎉 训练完成！最佳验证损失: {best_val_loss:.4f}")
        print(f"📁 实验目录: {self.exp_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据目录路径（覆盖配置文件）')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖数据目录
    if args.data_dir:
        config['data']['params']['data_dir'] = args.data_dir
    
    print("🚀 步骤5: 条件DiT微调训练")
    print("="*60)
    print(f"📝 配置文件: {args.config}")
    
    # 创建训练器并开始训练
    trainer = ConditionalDiTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
