#!/usr/bin/env python
"""
VA-VAE Micro-Doppler Semantic Alignment Fine-tuning
Production-ready script integrated with LightningDiT framework
"""

import os
import sys
import argparse
import datetime
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# 添加LightningDiT路径
if os.path.exists('/kaggle/working'):
    # Kaggle环境
    sys.path.insert(0, '/kaggle/working/VA-VAE/LightningDiT/vavae')
    sys.path.insert(0, '/kaggle/working/VA-VAE/LightningDiT')
else:
    # 本地环境
    vavae_path = Path(__file__).parent / 'LightningDiT' / 'vavae'
    sys.path.insert(0, str(vavae_path))
    sys.path.insert(0, str(vavae_path.parent))

# 导入原项目模块
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator
from ldm.util import instantiate_from_config


class MicroDopplerDataset(Dataset):
    """微多普勒数据集"""
    def __init__(self, data_root='/kaggle/input/dataset', split='train', 
                 val_users=[1, 10, 20, 30], img_size=256):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.samples = []
        
        for user_id in range(1, 32):
            user_folder = self.data_root / f"ID_{user_id}"
            if not user_folder.exists():
                continue
            
            is_val = user_id in val_users
            if (split == 'val' and is_val) or (split == 'train' and not is_val):
                for img_path in user_folder.glob("*.jpg"):
                    self.samples.append({
                        'path': str(img_path),
                        'user_id': user_id - 1  # 0-30
                    })
        
        print(f"{split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return {'image': img, 'user_id': sample['user_id']}


class UserContrastiveLoss(nn.Module):
    """用户对比损失 - 增强用户特征区分"""
    def __init__(self, temperature=0.1, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, z, user_ids):
        # 池化并归一化
        z_pooled = F.adaptive_avg_pool2d(z, 1).squeeze(-1).squeeze(-1)
        z_norm = F.normalize(z_pooled, dim=1)
        
        # 相似度矩阵
        sim = torch.matmul(z_norm, z_norm.T) / self.temperature
        
        # 用户掩码
        user_mask = user_ids.unsqueeze(0) == user_ids.unsqueeze(1)
        pos_mask = user_mask.float() - torch.eye(len(user_ids), device=z.device)
        neg_mask = 1 - user_mask.float()
        
        # 确保有正负样本
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        # 对比损失
        pos_sim = (sim * pos_mask).sum(1) / (pos_mask.sum(1) + 1e-8)
        neg_sim = (sim * neg_mask).sum(1) / (neg_mask.sum(1) + 1e-8)
        loss = F.relu(neg_sim - pos_sim + self.margin).mean()
        
        return loss


class VAVAEMicroDoppler(AutoencoderKL):
    """扩展的VA-VAE模型 - 添加用户对比学习"""
    
    def __init__(self, *args, **kwargs):
        # 提取自定义参数
        self.use_user_contrastive = kwargs.pop('use_user_contrastive', True)
        self.user_loss_weight = kwargs.pop('user_loss_weight', 0.5)
        self.stage_epochs = kwargs.pop('stage_epochs', [30, 20])
        
        super().__init__(*args, **kwargs)
        
        # 添加用户对比损失
        if self.use_user_contrastive:
            self.user_contrastive = UserContrastiveLoss(temperature=0.1, margin=0.5)
        
        # 训练阶段控制
        self.current_stage = 1
        self.stage1_epochs = self.stage_epochs[0]
        
    def training_step(self, batch, batch_idx):
        """重写训练步骤以添加用户损失"""
        inputs = self.get_input(batch, self.image_key)
        user_ids = batch.get('user_id', None)
        
        # 前向传播
        reconstructions, posterior, z, aux_feature = self(inputs)
        
        # 获取优化器
        opt_ae, opt_disc = self.optimizers()
        
        # 自动编码器损失
        enc_last_layer = self.encoder.conv_out.weight
        aeloss, log_dict_ae = self.loss(
            inputs, reconstructions, posterior, 0, self.global_step,
            last_layer=self.get_last_layer(), split="train", 
            z=z, aux_feature=aux_feature, enc_last_layer=enc_last_layer
        )
        
        # 添加用户对比损失
        if self.use_user_contrastive and user_ids is not None:
            user_loss = self.user_contrastive(z, user_ids)
            # 根据阶段调整权重
            if self.current_epoch < self.stage1_epochs:
                user_weight = self.user_loss_weight  # Stage 1: 0.5
            else:
                user_weight = self.user_loss_weight * 0.6  # Stage 2: 0.3
            aeloss = aeloss + user_weight * user_loss
            self.log("train/user_loss", user_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log("train/aeloss", aeloss, prog_bar=True, on_step=False, on_epoch=True)
        
        # 更新自动编码器
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()
        
        # 判别器损失
        discloss, log_dict_disc = self.loss(
            inputs, reconstructions, posterior, 1, self.global_step,
            last_layer=self.get_last_layer(), split="train", 
            enc_last_layer=enc_last_layer
        )
        
        self.log("train/discloss", discloss, prog_bar=True, on_step=False, on_epoch=True)
        
        # 更新判别器
        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()
        
    def on_train_epoch_start(self):
        """在每个epoch开始时检查是否切换阶段"""
        super().on_train_epoch_start()
        
        if self.current_epoch == self.stage1_epochs and self.current_stage == 1:
            print(f"\n>>> Switching to Stage 2 at epoch {self.current_epoch}")
            self.current_stage = 2
            
            # 调整损失权重
            if hasattr(self.loss, 'vf_weight'):
                self.loss.vf_weight = 0.1  # 降低VFM权重
            if hasattr(self.loss, 'discriminator_iter_start'):
                self.loss.discriminator_iter_start = 1  # 启用判别器
                
    def configure_optimizers(self):
        """配置优化器 - 使用保守学习率"""
        lr = 1e-5  # 微调使用更小的学习率
        
        # 自动编码器参数
        params = (list(self.encoder.parameters()) +
                 list(self.decoder.parameters()) +
                 list(self.quant_conv.parameters()) +
                 list(self.post_quant_conv.parameters()))
        
        if hasattr(self, 'linear_proj') and not self.proj_fix:
            params += list(self.linear_proj.parameters())
        
        # 冻结早期层（可选）
        for name, param in self.encoder.named_parameters():
            if 'down.0' in name or 'down.1' in name:
                param.requires_grad = False
        
        opt_ae = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        opt_disc = torch.optim.AdamW(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.9, 0.999), weight_decay=0.01
        )
        
        return [opt_ae, opt_disc], []


def create_model_config():
    """创建模型配置"""
    return OmegaConf.create({
        'target': 'train_vavae_microdoppler_final.VAVAEMicroDoppler',
        'params': {
            'embed_dim': 32,
            'use_vf': 'dinov2',
            'ckpt_path': '/kaggle/input/vavae-pretrained/vavae-imagenet256-f16d32-dinov2.pt',
            'use_user_contrastive': True,
            'user_loss_weight': 0.5,
            'stage_epochs': [30, 20],
            
            'ddconfig': {
                'double_z': False,
                'z_channels': 32,
                'resolution': 256,
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 1, 2, 2, 4],
                'num_res_blocks': 2,
                'attn_resolutions': [],
                'dropout': 0.0
            },
            
            'lossconfig': {
                'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                'params': {
                    'disc_start': 10000,  # Stage 1: 禁用判别器
                    'disc_num_layers': 3,
                    'disc_factor': 1.0,
                    'disc_weight': 1.0,
                    'pixelloss_weight': 1.0,
                    'perceptual_weight': 0.0,
                    'kl_weight': 1e-6,
                    'vf_weight': 0.3,  # Stage 1: 语义对齐
                    'distmat_weight': 1.0,
                    'cos_weight': 1.0,
                    'distmat_margin': 0.0,
                    'cos_margin': 0.0
                }
            }
        }
    })


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='/kaggle/input/dataset')
    parser.add_argument('--ckpt_path', type=str, 
                       default='/kaggle/input/vavae-pretrained/vavae-imagenet256-f16d32-dinov2.pt')
    args = parser.parse_args()
    
    # 设置种子
    seed_everything(args.seed)
    
    # 创建数据集
    train_dataset = MicroDopplerDataset(args.data_root, 'train')
    val_dataset = MicroDopplerDataset(args.data_root, 'val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 创建模型
    config = create_model_config()
    config.params.ckpt_path = args.ckpt_path
    model = instantiate_from_config(config)
    
    # 训练器配置
    callbacks = [
        ModelCheckpoint(
            dirpath='./checkpoints',
            filename='vavae-md-{epoch:02d}-{val_rec_loss:.4f}',
            monitor='val/rec_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 检测GPU
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if gpu_count > 0 else 'cpu',
        devices='auto',
        strategy=DDPStrategy(find_unused_parameters=False) if gpu_count > 1 else 'auto',
        precision=16,
        accumulate_grad_batches=2,  # 有效批次=16
        gradient_clip_val=1.0,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_progress_bar=True
    )
    
    # 训练信息
    print("\n" + "="*60)
    print("VA-VAE 微多普勒语义对齐微调")
    print("="*60)
    print(f"数据集: {args.data_root}")
    print(f"批次大小: {args.batch_size} x 2 (梯度累积) = 16")
    print(f"训练轮数: {args.epochs} (Stage1: 30, Stage2: 20)")
    print(f"预训练模型: {args.ckpt_path}")
    print("="*60 + "\n")
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    # 保存最终模型
    final_path = 'vavae_microdoppler_final.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'config': OmegaConf.to_container(config),
        'user_count': 31
    }, final_path)
    
    print(f"\n训练完成! 模型已保存到: {final_path}")
    print("\n下一步:")
    print("1. 下载微调后的模型")
    print("2. 使用step5_conditional_dit_training.py进行条件扩散模型训练")


if __name__ == '__main__':
    main()
